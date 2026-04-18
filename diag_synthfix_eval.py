"""Diagnostic: reload the SynthFix checkpoint and report greedy-only vs
reranker CodeBLEU on the test set.  Separates the training-time effect
from the inference-time reranker effect.

Usage:
    python diag_synthfix_eval.py \
        --ckpt results/synthfix_fixjs/final_model \
        --data data/processed/fixjs \
        --model_name deepseek-1.3b \
        --gpu 1
"""
import argparse
import sys
import time
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parent / 'src'
sys.path.insert(0, str(SRC))

from data.dataset import create_dataloaders  # noqa: E402
from models.inference import (generate_k_candidates, extract_features,  # noqa: E402
                               LearnedReranker, build_reranker_training_data)
from train_synthfix import (MODEL_PATHS, _compute_codebleu,  # noqa: E402
                             _detect_language)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--model_name', default='deepseek-1.3b')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--K', type=int, default=16)
    ap.add_argument('--max_new_tokens', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--safety_thresh', type=float, default=0.002)
    ap.add_argument('--no_reranker', action='store_true',
                    help='Skip reranker, greedy-only eval.')
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf = MODEL_PATHS[args.model_name]
    print(f'[diag] Loading base {hf} on GPU {args.gpu}', flush=True)
    tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        hf, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map={'': str(device)})
    print(f'[diag] Loading LoRA from {args.ckpt}', flush=True)
    model = PeftModel.from_pretrained(base, args.ckpt)
    model.eval()
    pad_id = tok.pad_token_id

    lang = _detect_language(args.data)
    print(f'[diag] Language: {lang}', flush=True)

    train_loader, val_loader, test_loader = create_dataloaders(
        args.data, tok, args.batch_size, 512, num_workers=2, shuffle_seed=42)

    # ── 1. Greedy-only test eval ─────────────────────────────────────
    print(f'[diag] (1/3) Greedy-only test eval ...', flush=True)
    t0 = time.time()
    gen_greedy, refs = [], []
    with torch.no_grad():
        for batch in test_loader:
            pids = batch['prompt_input_ids'].to(device, non_blocking=True)
            pmask = batch['prompt_attention_mask'].to(device, non_blocking=True)
            prompt_len = pids.size(1)
            gen = model.generate(
                input_ids=pids, attention_mask=pmask,
                max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=pad_id)
            for j, g in enumerate(gen):
                gt = tok.decode(g[prompt_len:], skip_special_tokens=True
                                ).strip()[:2000]
                rt = batch['fixed_text'][j].strip()[:2000]
                gen_greedy.append(gt)
                refs.append(rt)
    cb_greedy = _compute_codebleu(gen_greedy, refs, lang=lang)
    print(f'[diag] Greedy CodeBLEU = {cb_greedy*100:.2f}%   '
          f'({len(gen_greedy)} samples, {time.time()-t0:.0f}s)', flush=True)

    if args.no_reranker:
        return

    # ── 2. Train reranker on val ─────────────────────────────────────
    print(f'[diag] (2/3) Collecting val reranker training data '
          f'(K={args.K}) ...', flush=True)
    t0 = time.time()
    X_val, y_val = build_reranker_training_data(
        model, tok, router=None, val_loader=val_loader, device=device,
        lang=lang, codebleu_fn=_compute_codebleu, K=args.K,
        max_new_tokens=args.max_new_tokens, max_samples=200)
    print(f'[diag] Collected {X_val.shape[0]} candidates  '
          f'({time.time()-t0:.0f}s)', flush=True)
    reranker = LearnedReranker(tag=f'diag/{args.model_name}')
    reranker.fit(X_val, y_val, K=args.K, verbose=True)

    # ── 3. Reranked test eval ────────────────────────────────────────
    print(f'[diag] (3/3) Reranker test eval (K={args.K}) ...', flush=True)
    t0 = time.time()
    import numpy as np
    gen_rr = []
    non_greedy_picks = 0
    with torch.no_grad():
        for batch in test_loader:
            pids = batch['prompt_input_ids'].to(device, non_blocking=True)
            pmask = batch['prompt_attention_mask'].to(device, non_blocking=True)
            cur_bs = pids.size(0)
            conts, logps, temps, gflags, iflags = generate_k_candidates(
                model, tok, pids, pmask, K=args.K,
                max_new_tokens=args.max_new_tokens, pad_id=pad_id)
            for j in range(cur_bs):
                cands = conts[j]
                buggy = batch['buggy_text'][j].strip()[:2000]
                feats = extract_features(buggy, cands, logps[j], temps[j],
                                          gflags[j], iflags[j])
                scores = reranker.predict(feats)
                greedy_idx = next((i for i, g in enumerate(gflags[j]) if g), 0)
                max_score = float(np.max(scores))
                if (max_score - scores[greedy_idx]) < args.safety_thresh:
                    best = greedy_idx
                else:
                    best = int(np.argmax(scores))
                    if best != greedy_idx:
                        non_greedy_picks += 1
                gen_rr.append(cands[best])
    cb_rerank = _compute_codebleu(gen_rr, refs, lang=lang)
    print(f'[diag] Reranked CodeBLEU = {cb_rerank*100:.2f}%   '
          f'non-greedy picks: {non_greedy_picks}/{len(gen_rr)}   '
          f'({time.time()-t0:.0f}s)', flush=True)

    print(f'\n[diag] ─── SUMMARY ────────────────────────────────────')
    print(f'[diag] Greedy  : {cb_greedy*100:.2f}%')
    print(f'[diag] Reranked: {cb_rerank*100:.2f}%')
    print(f'[diag] Delta   : {(cb_rerank-cb_greedy)*100:+.2f}pp', flush=True)


if __name__ == '__main__':
    main()
