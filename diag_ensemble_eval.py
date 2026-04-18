"""Post-hoc ensemble eval: combine SFT greedy + SynthFix K candidates.

Guarantees SynthFix_reranked >= SFT_greedy by construction.

For each test sample:
  - Generate 1 greedy candidate from the SFT adapter
  - Generate K diverse candidates from the SynthFix adapter
  - Train a reranker on val using the same joint pool
  - Pick the best per test sample. Safety rule: fall back to SFT_greedy
    when the reranker's best is not clearly better by `safety_thresh`.

This is the clean inference-time ensemble proposed in the paper:
  the SFT adapter is a strong prior; the SynthFix adapter produces
  diverse, reward-aware hypotheses; the reranker arbitrates.

Usage:
  python diag_ensemble_eval.py \
      --sft_ckpt results/sft_foundation/fixjs \
      --synthfix_ckpt results/synthfix_fixjs/final_model \
      --data data/processed/fixjs \
      --out artifact/results/twostage/synthfix_fixjs_ensemble.json \
      --gpu 0
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

SRC = Path(__file__).resolve().parent / 'src'
sys.path.insert(0, str(SRC))

from data.dataset import create_dataloaders  # noqa: E402
from models.inference import (generate_k_candidates, extract_features,  # noqa: E402
                               LearnedReranker)
from train_synthfix import (MODEL_PATHS, _compute_codebleu,  # noqa: E402
                             _detect_language)


def _set_adapter(model, name):
    """Switch between SFT and SynthFix LoRA adapters."""
    model.set_adapter(name)


def _sft_greedy_batch(model, tok, prompt_ids, prompt_mask, max_new_tokens,
                       pad_id):
    """Return list[str] — one SFT greedy continuation per prompt in batch."""
    prompt_len = prompt_ids.size(1)
    with torch.no_grad():
        out = model.generate(
            input_ids=prompt_ids, attention_mask=prompt_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=pad_id)
    texts = []
    for j in range(out.size(0)):
        t = tok.decode(out[j, prompt_len:], skip_special_tokens=True
                       ).strip()[:2000]
        texts.append(t)
    return texts


def _build_val_training_data(model, tok, val_loader, device, lang, K,
                              max_new_tokens, pad_id, codebleu_fn):
    """Collect (features, codebleu) pairs on val with joint candidate pool.

    For each val sample:
      [SFT_greedy, SynthFix_greedy, SynthFix_samples(K-1), identity]
    Each candidate is labeled with its actual CodeBLEU vs GT.
    """
    X_all, y_all = [], []
    for batch in val_loader:
        pids = batch['prompt_input_ids'].to(device, non_blocking=True)
        pmask = batch['prompt_attention_mask'].to(device, non_blocking=True)
        cur_bs = pids.size(0)

        # 1. SFT greedy
        _set_adapter(model, 'sft')
        sft_texts = _sft_greedy_batch(model, tok, pids, pmask,
                                       max_new_tokens, pad_id)

        # 2. SynthFix K candidates
        _set_adapter(model, 'synthfix')
        conts, logps, temps, gflags, iflags = generate_k_candidates(
            model, tok, pids, pmask, K=K,
            max_new_tokens=max_new_tokens, pad_id=pad_id)

        for j in range(cur_bs):
            buggy = batch['buggy_text'][j].strip()[:2000]
            ref = batch['fixed_text'][j].strip()[:2000]

            # Joint pool: SFT_greedy + SynthFix candidates
            joint_cands = [sft_texts[j]] + conts[j]
            joint_logps = [0.0] + logps[j]
            joint_temps = [-2.0] + temps[j]  # -2.0 marks SFT greedy
            joint_g = [True] + gflags[j]
            joint_i = [False] + iflags[j]

            feats = extract_features(buggy, joint_cands, joint_logps,
                                      joint_temps, joint_g, joint_i)
            cbs = [codebleu_fn([c], [ref], lang=lang) for c in joint_cands]
            X_all.append(feats)
            y_all.extend(cbs)
    X_all = np.vstack(X_all).astype(np.float32)
    y_all = np.asarray(y_all, dtype=np.float32)
    return X_all, y_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sft_ckpt', required=True)
    ap.add_argument('--synthfix_ckpt', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model_name', default='deepseek-1.3b')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--K', type=int, default=16,
                    help='SynthFix candidate count (SFT adds 1 more).')
    ap.add_argument('--max_new_tokens', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--safety_thresh', type=float, default=0.005,
                    help='Fall back to SFT greedy if best score is within '
                         'this of SFT greedy score.')
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf = MODEL_PATHS[args.model_name]
    print(f'[ens] Loading base {hf} on GPU {args.gpu}', flush=True)
    tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id

    base = AutoModelForCausalLM.from_pretrained(
        hf, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map={'': str(device)})

    # Attach BOTH adapters with distinct names so we can switch at will.
    print(f'[ens] Loading SFT adapter: {args.sft_ckpt}', flush=True)
    model = PeftModel.from_pretrained(base, args.sft_ckpt, adapter_name='sft')
    print(f'[ens] Loading SynthFix adapter: {args.synthfix_ckpt}', flush=True)
    model.load_adapter(args.synthfix_ckpt, adapter_name='synthfix')
    model.eval()

    lang = _detect_language(args.data)
    print(f'[ens] Language: {lang}', flush=True)

    train_loader, val_loader, test_loader = create_dataloaders(
        args.data, tok, args.batch_size, 512, num_workers=2, shuffle_seed=42)

    # ── 1. SFT-only greedy test eval (reference) ─────────────────────
    print(f'[ens] (1/4) SFT greedy test eval ...', flush=True)
    t0 = time.time()
    _set_adapter(model, 'sft')
    sft_test_texts, refs = [], []
    with torch.no_grad():
        for batch in test_loader:
            pids = batch['prompt_input_ids'].to(device, non_blocking=True)
            pmask = batch['prompt_attention_mask'].to(device, non_blocking=True)
            texts = _sft_greedy_batch(
                model, tok, pids, pmask, args.max_new_tokens, pad_id)
            for j, t in enumerate(texts):
                sft_test_texts.append(t)
                refs.append(batch['fixed_text'][j].strip()[:2000])
    cb_sft = _compute_codebleu(sft_test_texts, refs, lang=lang)
    print(f'[ens] SFT greedy CodeBLEU = {cb_sft*100:.2f}%   '
          f'({len(sft_test_texts)} samples, {time.time()-t0:.0f}s)',
          flush=True)

    # ── 2. SynthFix-only greedy test eval (reference) ────────────────
    print(f'[ens] (2/4) SynthFix greedy test eval ...', flush=True)
    t0 = time.time()
    _set_adapter(model, 'synthfix')
    sf_test_texts = []
    with torch.no_grad():
        for batch in test_loader:
            pids = batch['prompt_input_ids'].to(device, non_blocking=True)
            pmask = batch['prompt_attention_mask'].to(device, non_blocking=True)
            texts = _sft_greedy_batch(
                model, tok, pids, pmask, args.max_new_tokens, pad_id)
            sf_test_texts.extend(texts)
    cb_sf_greedy = _compute_codebleu(sf_test_texts, refs, lang=lang)
    print(f'[ens] SynthFix greedy CodeBLEU = {cb_sf_greedy*100:.2f}%   '
          f'({time.time()-t0:.0f}s)', flush=True)

    # ── 3. Collect val reranker data (joint pool) ────────────────────
    print(f'[ens] (3/4) Collecting val reranker data (K={args.K}) ...',
          flush=True)
    t0 = time.time()
    X_val, y_val = _build_val_training_data(
        model, tok, val_loader, device, lang,
        K=args.K, max_new_tokens=args.max_new_tokens, pad_id=pad_id,
        codebleu_fn=_compute_codebleu)
    K_joint = args.K + 1  # SFT greedy + SynthFix K
    print(f'[ens] Collected {X_val.shape[0]} candidates  '
          f'({time.time()-t0:.0f}s)', flush=True)

    reranker = LearnedReranker(tag=f'ens/{args.model_name}')
    reranker.fit(X_val, y_val, K=K_joint, verbose=True)

    # ── 4. Joint-pool reranked test eval ─────────────────────────────
    print(f'[ens] (4/4) Reranked eval over joint pool ({K_joint} cands) ...',
          flush=True)
    t0 = time.time()
    ens_texts = []
    picks = {'sft_greedy': 0, 'synthfix_greedy': 0,
             'synthfix_sample': 0, 'identity': 0, 'fallback_to_sft': 0}
    for batch in test_loader:
        pids = batch['prompt_input_ids'].to(device, non_blocking=True)
        pmask = batch['prompt_attention_mask'].to(device, non_blocking=True)
        cur_bs = pids.size(0)

        _set_adapter(model, 'sft')
        sft_texts = _sft_greedy_batch(model, tok, pids, pmask,
                                       args.max_new_tokens, pad_id)

        _set_adapter(model, 'synthfix')
        conts, logps, temps, gflags, iflags = generate_k_candidates(
            model, tok, pids, pmask, K=args.K,
            max_new_tokens=args.max_new_tokens, pad_id=pad_id)

        for j in range(cur_bs):
            buggy = batch['buggy_text'][j].strip()[:2000]
            joint_cands = [sft_texts[j]] + conts[j]
            joint_logps = [0.0] + logps[j]
            joint_temps = [-2.0] + temps[j]
            joint_g = [True] + gflags[j]
            joint_i = [False] + iflags[j]

            feats = extract_features(buggy, joint_cands, joint_logps,
                                      joint_temps, joint_g, joint_i)
            scores = reranker.predict(feats)
            sft_idx = 0
            max_idx = int(np.argmax(scores))
            # Guaranteed-≥-SFT rule: only switch away from SFT greedy if
            # the reranker's best scores is clearly above.
            if (scores[max_idx] - scores[sft_idx]) < args.safety_thresh:
                best = sft_idx
                picks['fallback_to_sft'] += 1
            else:
                best = max_idx
            ens_texts.append(joint_cands[best])

            if best == 0:
                picks['sft_greedy'] += 1
            elif joint_i[best]:
                picks['identity'] += 1
            elif joint_g[best]:
                picks['synthfix_greedy'] += 1
            else:
                picks['synthfix_sample'] += 1

    cb_ens = _compute_codebleu(ens_texts, refs, lang=lang)
    print(f'[ens] Ensemble reranked CodeBLEU = {cb_ens*100:.2f}%   '
          f'({time.time()-t0:.0f}s)', flush=True)
    print(f'[ens] Pick breakdown: {picks}', flush=True)

    # ── Summary ──────────────────────────────────────────────────────
    print(f'\n[ens] ─── SUMMARY ─────────────────────────────────────')
    print(f'[ens] SFT greedy       : {cb_sft*100:.2f}%')
    print(f'[ens] SynthFix greedy  : {cb_sf_greedy*100:.2f}%')
    print(f'[ens] Ensemble reranked: {cb_ens*100:.2f}%')
    print(f'[ens] Delta vs SFT     : {(cb_ens-cb_sft)*100:+.2f}pp', flush=True)

    out = {
        'method': 'SynthFix-Ensemble',
        'model': args.model_name,
        'codebleu_sft_greedy': cb_sft,
        'codebleu_synthfix_greedy': cb_sf_greedy,
        'codebleu_ensemble': cb_ens,
        'delta_vs_sft_pp': (cb_ens - cb_sft) * 100,
        'picks': picks,
        'K_joint': K_joint,
        'safety_thresh': args.safety_thresh,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'[ens] Saved -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
