#!/usr/bin/env python3
"""
SynthFix: Full Experiment Suite — Overnight Runner

Trains SFT / RFT / SynthFix across multiple model sizes and datasets.
Uses both A6000 GPUs in parallel where possible.

Hyperparameters matched to the proven rebuttal configuration:
  - lr=2e-4, epochs=3, LoRA r=16, alpha=32
  - Per-sample routing with separate baselines

Datasets:
  - FixJS      (JavaScript)  — 2000 train / 200 test (sampled)
  - CodeFlaws  (C)           — full (~3100 train / 391 test)
  - SVEN       (Python)      — full (~716 train / 42 test)

Models (ascending size):
  1. deepseek-1.3b  (1.3B)
  2. starcoder2-3b  (3B)
  3. codellama-7b   (7B)
  4. deepseek-6.7b  (6.7B)
  5. starcoder2-7b  (7B)

Usage:
    nohup python run_all_experiments.py > results/experiments.log 2>&1 &
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

import types as _types

if 'torch._dynamo' not in sys.modules:
    _m = _types.ModuleType('torch._dynamo')
    _m.disable = lambda fn=None, recursive=True: fn if fn else (lambda f: f)
    _m.graph_break = lambda: None
    _m.is_compiling = lambda: False
    _mc = _types.ModuleType('torch._dynamo.config')
    _mc.suppress_errors = True
    _m.config = _mc
    sys.modules['torch._dynamo'] = _m
    sys.modules['torch._dynamo.config'] = _mc

# ── Configuration ───────────────────────────────────────────────────────

RAW_DATA = Path(os.environ.get(
    'SYNTHFIX_DATA',
    str(Path(__file__).resolve().parent.parent / 'data' / 'raw_benchmarks')
))

DATASETS = {
    'fixjs': {
        'raw_dir': str(RAW_DATA / 'fixjs'),
        'language': 'javascript',
        'train_cap': 2000,
        'val_cap': 100,
        'test_cap': 200,
    },
    'codeflaws': {
        'raw_dir': str(RAW_DATA / 'codeflaws'),
        'language': 'c',
        'train_cap': None,
        'val_cap': None,
        'test_cap': None,
    },
    'sven': {
        'raw_dir': str(RAW_DATA / 'sven'),
        'language': 'python',
        'train_cap': None,
        'val_cap': None,
        'test_cap': None,
    },
}

# All models use rebuttal-matched config: lr=2e-4, epochs=4, LoRA r=16
# (4 epochs: 1 SFT warmup + 3 curriculum for SynthFix; fair comparison)
MODELS = [
    {
        'model': 'deepseek-1.3b',
        'batch_size': 16,
        'epochs': 4,
        'lora_rank': 16,
        'grad_checkpoint': False,
        'max_new_tokens': 128,
        'lr': 2e-4,
    },
    {
        'model': 'starcoder2-3b',
        'batch_size': 8,
        'epochs': 4,
        'lora_rank': 16,
        'grad_checkpoint': False,
        'max_new_tokens': 128,
        'lr': 2e-4,
    },
    {
        'model': 'codellama-7b',
        'batch_size': 4,
        'epochs': 4,
        'lora_rank': 16,
        'grad_checkpoint': True,
        'max_new_tokens': 128,
        'lr': 2e-4,
    },
    {
        'model': 'deepseek-6.7b',
        'batch_size': 4,
        'epochs': 4,
        'lora_rank': 16,
        'grad_checkpoint': True,
        'max_new_tokens': 128,
        'lr': 2e-4,
    },
    {
        'model': 'starcoder2-7b',
        'batch_size': 4,
        'epochs': 4,
        'lora_rank': 16,
        'grad_checkpoint': True,
        'max_new_tokens': 128,
        'lr': 2e-4,
    },
]

NUM_WORKERS = 4

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / 'src'
RESULTS_DIR = SCRIPT_DIR / 'results'

_log_file = None


def _log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    if _log_file:
        _log_file.write(line + '\n')
        _log_file.flush()


# ── Data Preparation ────────────────────────────────────────────────────

def prepare_all_data():
    """Process all three datasets, apply sample caps, return dict of paths."""
    sys.path.insert(0, str(SRC_DIR))
    from data.process_benchmarks import process_fixjs, process_codeflaws, process_sven

    tmpbase = Path(tempfile.mkdtemp(prefix='synthfix_data_'))
    data_dirs = {}

    processors = {
        'fixjs': process_fixjs,
        'codeflaws': process_codeflaws,
        'sven': process_sven,
    }

    for ds_name, ds_cfg in DATASETS.items():
        _log(f'Processing {ds_name}...')
        proc = processors[ds_name]
        proc(ds_cfg['raw_dir'], tmpbase)
        data_dir = tmpbase / ds_name

        for split_name, cap_key in [('train', 'train_cap'),
                                     ('val', 'val_cap'),
                                     ('test', 'test_cap')]:
            cap = ds_cfg.get(cap_key)
            if cap is not None:
                f = data_dir / f'{split_name}.json'
                data = json.loads(f.read_text())
                if len(data) > cap:
                    data = data[:cap]
                    f.write_text(json.dumps(data))

        for split in ['train', 'val', 'test']:
            f = data_dir / f'{split}.json'
            n = len(json.loads(f.read_text()))
            _log(f'  {ds_name}/{split}: {n} samples')

        data_dirs[ds_name] = str(data_dir)

    return data_dirs


# ── Worker Process ──────────────────────────────────────────────────────

def run_worker(args):
    """Subprocess worker: train one method on one GPU."""
    import numpy as np
    import torch
    torch.backends.cudnn.benchmark = True

    sys.path.insert(0, str(SRC_DIR))
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from data.dataset import create_dataloaders
    from models.reward import compute_reward, _try_parse_ast, _chrf_similarity
    from models.router import (RouterModel, compute_batch_features,
                               normalize_features)
    from models.inference import (generate_k_candidates, extract_features,
                                   LearnedReranker,
                                   compute_budget_for_prob_hard,
                                   build_reranker_training_data,
                                   NUM_FEATURES)
    from train_synthfix import (get_lora_config, MODEL_PATHS,
                                _compute_codebleu,
                                _detect_language)

    method = args.method
    gpu = args.gpu
    device = torch.device(f'cuda:{gpu}')
    model_name = args.model_name
    hf_path = MODEL_PATHS[model_name]
    bs = args.batch_size
    epochs = args.epochs
    lr = args.lr
    lora_rank = args.lora_rank
    max_new_tokens = args.max_new_tokens
    dataset_name = args.dataset_name

    lang = _detect_language(args.data_dir)

    seed = args.seed
    import random, numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    P = lambda *a, **kw: print(*a, **kw, flush=True)
    tag = f'{method.upper()}/{model_name}/{dataset_name}'

    P(f'[{tag}] Loading {hf_path} on GPU {gpu} (seed={seed})...')

    tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_path, torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={'': str(device)})
    model = get_peft_model(model, get_lora_config(model_name, lora_rank))

    # Two-stage bootstrap: if caller provides an SFT LoRA ckpt, reload
    # the LoRA adapter weights from disk so RFT/SynthFix starts from a
    # strong SFT checkpoint (not a random LoRA init).
    if getattr(args, 'init_from_ckpt', None):
        from peft import PeftModel
        P(f'[{tag}] Initialising LoRA from {args.init_from_ckpt}')
        base = model.base_model.model
        del model
        torch.cuda.empty_cache()
        model = PeftModel.from_pretrained(
            base, args.init_from_ckpt, is_trainable=True)
        model = model.to(device)

    if args.grad_checkpoint:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    P(f'[{tag}] LoRA r={lora_rank}  Params: {trainable:,}/{total_p:,}')
    P(f'[{tag}] BS={bs}  epochs={epochs}  lr={lr}  '
      f'grad_ckpt={args.grad_checkpoint}')

    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir, tokenizer, bs, 512, num_workers=NUM_WORKERS,
        shuffle_seed=seed)
    pad_id = tokenizer.pad_token_id
    n_batches = len(train_loader)
    P(f'[{tag}] {n_batches} batches/epoch')

    t0 = time.time()

    # ── SFT ─────────────────────────────────────────────────────────
    # Uses best-validation-loss checkpoint selection so the baseline is
    # not penalized by overfitting on small training sets. SynthFix uses
    # the same selection strategy — fair comparison.
    if method == 'sft':
        import tempfile as _tmp
        _ckpt_root = _tmp.mkdtemp(prefix='sft_ckpt_')
        best_val = float('inf')
        best_dir = None
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        for epoch in range(epochs):
            model.train()
            eloss = 0
            for batch in train_loader:
                ids = batch['input_ids'].to(device, non_blocking=True)
                mask = batch['attention_mask'].to(device, non_blocking=True)
                lab = batch['labels'].to(device, non_blocking=True)
                out = model(input_ids=ids, attention_mask=mask, labels=lab)
                opt.zero_grad()
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                eloss += out.loss.item()
            # Validation for early-stopping selection
            model.eval()
            vloss = vn = 0
            with torch.no_grad():
                for vb in val_loader:
                    ids = vb['input_ids'].to(device, non_blocking=True)
                    mask = vb['attention_mask'].to(device, non_blocking=True)
                    lab = vb['labels'].to(device, non_blocking=True)
                    vo = model(input_ids=ids, attention_mask=mask, labels=lab)
                    vloss += vo.loss.item(); vn += 1
            vloss /= max(vn, 1)
            P(f'  [{tag}] Epoch {epoch+1}/{epochs}: '
              f'loss={eloss/n_batches:.4f}  val_loss={vloss:.4f}')
            if vloss < best_val:
                best_val = vloss
                best_dir = os.path.join(_ckpt_root, f'ep{epoch+1}')
                model.save_pretrained(best_dir)
        # Reload best checkpoint
        if best_dir is not None:
            from peft import PeftModel
            P(f'  [{tag}] Reloading best (val_loss={best_val:.4f})')
            model = model.base_model.model
            model = PeftModel.from_pretrained(model, best_dir)
            model = model.to(device)
            if getattr(args, 'save_ckpt_to', None):
                os.makedirs(args.save_ckpt_to, exist_ok=True)
                model.save_pretrained(args.save_ckpt_to)
                P(f'  [{tag}] Saved best SFT ckpt -> {args.save_ckpt_to}')

    # ── RFT (REINFORCE + baseline + SFT anchor) ─────────────────────
    # Proper implementation. Prior version had two catastrophic bugs:
    #   (1) Wrong sign on the policy gradient: `(-out.loss * adv)` —
    #       HF's loss IS -log pi, so this *maximizes* CE of high-reward
    #       samples → collapse.
    #   (2) No SFT anchor → unconstrained RL drifts off-distribution.
    # Fix: advantage-weighted log-prob loss (correct sign), moving
    # baseline, 1-epoch SFT warmup, and a mixture with SFT CE so the
    # policy stays close to the teacher.
    elif method == 'rft':
        import tempfile as _tmp
        _ckpt_root = _tmp.mkdtemp(prefix='rft_ckpt_')
        best_val = float('inf')
        best_dir = None
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr * 0.5, weight_decay=0.01)
        # Skip RFT's own SFT warmup when bootstrapped from an SFT ckpt.
        if getattr(args, 'init_from_ckpt', None):
            warmup_epochs = 0
            P(f'  [{tag}] Bootstrapped from SFT ckpt -> '
              f'skipping SFT warmup, going straight to REINFORCE')
        else:
            warmup_epochs = 1 if epochs > 1 else 0
        baseline = 0.0
        baseline_mom = 0.9
        BETA_RL = 0.5
        SAMPLE_TEMP = float(getattr(args, 'rft_rl_temp', 0.8))
        RL_NO_REPEAT = int(getattr(args, 'rft_no_repeat_ngram', 0) or 0)
        RL_MAX_GEN = min(128, max_new_tokens)

        for epoch in range(epochs):
            phase = 'SFT' if epoch < warmup_epochs else 'RFT'
            model.train()
            ep_loss = 0.0
            ep_reward = 0.0
            nb = 0
            for batch in train_loader:
                sft_ids = batch['input_ids'].to(device, non_blocking=True)
                sft_mask = batch['attention_mask'].to(device, non_blocking=True)
                sft_lab = batch['labels'].to(device, non_blocking=True)

                if phase == 'SFT':
                    opt.zero_grad()
                    out = model(input_ids=sft_ids, attention_mask=sft_mask,
                                labels=sft_lab)
                    loss = out.loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0)
                    opt.step()
                    ep_loss += loss.item()
                    nb += 1
                    continue
                else:
                    pids = batch['prompt_input_ids'].to(device, non_blocking=True)
                    pmask = batch['prompt_attention_mask'].to(device,
                                                               non_blocking=True)
                    cur_bs = pids.size(0)
                    prompt_len = pids.size(1)
                    with torch.no_grad():
                        _gen_kwargs = dict(
                            input_ids=pids, attention_mask=pmask,
                            max_new_tokens=RL_MAX_GEN, do_sample=True,
                            temperature=SAMPLE_TEMP, top_p=0.95,
                            pad_token_id=pad_id)
                        if RL_NO_REPEAT > 0:
                            _gen_kwargs['no_repeat_ngram_size'] = RL_NO_REPEAT
                        gen_ids = model.generate(**_gen_kwargs)

                    rewards = []
                    for j in range(cur_bs):
                        cont = tokenizer.decode(
                            gen_ids[j, prompt_len:], skip_special_tokens=True)
                        rewards.append(float(
                            compute_reward(cont, batch['fixed_text'][j])))
                    rewards_t = torch.tensor(rewards, device=device,
                                              dtype=torch.float32)
                    mean_r = rewards_t.mean().item()
                    baseline = (baseline_mom * baseline
                                + (1.0 - baseline_mom) * mean_r)
                    advantages = rewards_t - baseline
                    adv_std = advantages.std().clamp(min=1e-4)
                    advantages = advantages / adv_std

                    full_ids = gen_ids
                    full_mask = (full_ids != pad_id).long()
                    rl_labels = full_ids.clone()
                    rl_labels[:, :prompt_len] = -100
                    rl_labels[full_mask == 0] = -100

                    # Per-sample CE over generated tokens
                    out_rl = model(input_ids=full_ids, attention_mask=full_mask)
                    logits = out_rl.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = rl_labels[..., 1:].contiguous()
                    bs_, sl_m1, vocab = shift_logits.shape
                    per_tok = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, vocab),
                        shift_labels.view(-1),
                        reduction='none', ignore_index=-100,
                    ).view(bs_, sl_m1)
                    tok_mask = (shift_labels != -100).float()
                    n_tok = tok_mask.sum(1)
                    valid = (n_tok > 0).float()
                    per_sample_ce = (per_tok * tok_mask).sum(1) \
                                     / n_tok.clamp(min=1)

                    # loss = advantage × CE(sampled).
                    # CE = -log pi(gen|prompt). Minimizing adv*CE when
                    # adv > 0 → decrease CE → increase log pi of the
                    # high-reward sample. Correct sign.
                    rl_loss = (advantages * per_sample_ce * valid).sum() \
                              / valid.sum().clamp(min=1)
                    ep_reward += mean_r

                    # Split RL + SFT backward to keep peak memory low:
                    # one forward graph in memory at a time.
                    opt.zero_grad()
                    (BETA_RL * rl_loss).backward()
                    del out_rl, logits, per_tok, per_sample_ce, rl_loss
                    out_sft = model(input_ids=sft_ids, attention_mask=sft_mask,
                                    labels=sft_lab)
                    ((1.0 - BETA_RL) * out_sft.loss).backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0)
                    opt.step()
                    ep_loss += out_sft.loss.item()
                    nb += 1

            # Val loss (for selection)
            model.eval()
            v_loss = 0.0
            v_n = 0
            with torch.no_grad():
                for vb in val_loader:
                    v_ids = vb['input_ids'].to(device, non_blocking=True)
                    v_m = vb['attention_mask'].to(device, non_blocking=True)
                    v_l = vb['labels'].to(device, non_blocking=True)
                    vout = model(input_ids=v_ids, attention_mask=v_m,
                                 labels=v_l)
                    v_loss += vout.loss.item()
                    v_n += 1
            v_loss /= max(v_n, 1)
            P(f'  [{tag}] Epoch {epoch+1}/{epochs} [{phase}]: '
              f'loss={ep_loss/max(nb,1):.4f}  '
              f'reward={ep_reward/max(nb,1):.4f}  '
              f'baseline={baseline:.4f}  val_loss={v_loss:.4f}')
            if v_loss < best_val:
                best_val = v_loss
                best_dir = os.path.join(_ckpt_root, f'ep{epoch+1}')
                model.save_pretrained(best_dir)
                P(f'  [{tag}] -> New best (val_loss={v_loss:.4f})')

        if best_dir is not None:
            from peft import PeftModel
            P(f'  [{tag}] Reloading best (val_loss={best_val:.4f})')
            model = model.base_model.model
            model = PeftModel.from_pretrained(model, best_dir)
            model = model.to(device)

    # ── SynthFix (per-sample routing — matching rebuttal) ──────────
    elif method == 'synthfix':
        from train_synthfix import train_synthfix as _train_sf

        router = RouterModel(input_size=4, hidden_size=64).to(device)

        class _A:
            pass
        _a = _A()
        _a.lr = lr
        _a.epochs = epochs
        _a.lambda_ast = 0.333
        _a.lambda_cfg = 0.333
        _a.lambda_sem = 0.334
        _a.max_new_tokens = max_new_tokens
        _a.dataset = args.data_dir
        # v12: forward RL knobs if CLI provided them (None → module default).
        for _k in ('rl_beta', 'rloo_k', 'rl_temp', 'rl_top_p',
                   'rl_no_repeat_ngram'):
            _v = getattr(args, _k, None)
            if _v is not None:
                setattr(_a, _k, _v)
                P(f'  [{tag}] override {_k}={_v}')
        # Two-stage: if we start from an SFT ckpt, skip SynthFix's
        # internal SFT warmup and go straight to router-gated RLOO.
        if getattr(args, 'init_from_ckpt', None):
            _a.sft_warmup_epochs = 0
            P(f'  [{tag}] Bootstrapped from SFT ckpt -> '
              f'skipping SynthFix SFT warmup, going straight to RLOO')
        elif getattr(args, 'sft_warmup_epochs', -1) >= 0:
            _a.sft_warmup_epochs = args.sft_warmup_epochs

        # Persistent checkpoint dir for faster iteration. Keyed on
        # model + dataset so multiple experiments don't collide.
        ckpt_dir = str(Path(tempfile.gettempdir())
                       / f'synthfix_ckpt_{model_name}_{dataset_name}')
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        model, router = _train_sf(
            model, tokenizer, router,
            train_loader, val_loader,
            ckpt_dir, _a, device)

    train_time = time.time() - t0
    P(f'  [{tag}] Training: {train_time:.0f}s, evaluating...')

    # ── Reranker training (SynthFix only) ───────────────────────────
    # Label-side: for each val sample generate K=8 candidates, compute
    # real CodeBLEU of each vs GT, train a LightGBM regressor on the
    # reference-free 13-dim feature matrix. Used at test time to pick
    # the best candidate per input.
    reranker = None
    K_CANDS = int(getattr(args, 'num_rerank_cands', 16))
    RERANK_MARGIN = float(getattr(args, 'rerank_margin', 0.015))
    NO_RERANK = bool(getattr(args, 'no_rerank', False))
    if method == 'synthfix' and not NO_RERANK:
        P(f'  [{tag}] Training inference-time reranker on val set '
          f'(K={K_CANDS}, margin={RERANK_MARGIN})...')
        t_rr = time.time()
        X_val, y_val = build_reranker_training_data(
            model, tokenizer, router, val_loader, device, lang,
            codebleu_fn=_compute_codebleu,
            K=K_CANDS, max_new_tokens=max_new_tokens,
            max_samples=100,
        )
        P(f'  [{tag}]   collected {X_val.shape[0]} val candidates, '
          f'mean CodeBLEU={y_val.mean()*100:.2f}%  '
          f'({time.time()-t_rr:.0f}s)')
        reranker = LearnedReranker(tag=f'{model_name}/{dataset_name}')
        reranker.fit(X_val, y_val, K=K_CANDS, verbose=True)
    elif method == 'synthfix' and NO_RERANK:
        P(f'  [{tag}] Reranker DISABLED (--no_rerank) — '
          f'test-time = greedy only.')

    # ── Evaluate ────────────────────────────────────────────────────
    # Continuation-only eval: we decode ONLY the newly generated tokens
    # (everything after the prompt) and compare to the fixed reference.
    # This matches the original rebuttal evaluation and avoids the
    # prompt-echo inflation that was hiding method differences under the
    # prior "full-output" protocol.
    #
    # SynthFix uses router-gated inference-time best-of-K with a
    # learned LightGBM reranker. The SFT/RFT baselines use plain
    # greedy decoding.
    model.eval()
    em = total = 0
    all_gen, all_ref = [], []
    # For SynthFix: also track greedy-only outputs so we can report the
    # training-time effect separately from the reranker's contribution.
    all_greedy = []

    # Track compute stats for SynthFix (distribution of K)
    k_distribution = {}

    with torch.no_grad():
        for batch in test_loader:
            pids = batch['prompt_input_ids'].to(device, non_blocking=True)
            pmask = batch['prompt_attention_mask'].to(device, non_blocking=True)
            cur_bs = pids.size(0)
            prompt_pad_len = pids.size(1)

            if method != 'synthfix' or NO_RERANK:
                # Baselines (and --no_rerank ablation): plain greedy —
                # matches the greedy candidate used by the reranker
                # path so the ablation is clean (no ngram blocking).
                gen = model.generate(
                    input_ids=pids, attention_mask=pmask,
                    max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=pad_id)
                decoded = [tokenizer.decode(
                               g[prompt_pad_len:], skip_special_tokens=True
                           ).strip()[:2000]
                           for g in gen]
                if method == 'synthfix' and NO_RERANK:
                    # For the ablation path, still need all_greedy
                    # populated for consistent bookkeeping.
                    all_greedy.extend(decoded)
            else:
                # SynthFix best-of-K inference. K_CANDS across the batch
                # with a wide temperature schedule for candidate
                # diversity + learned LightGBM reranker.
                K_batch = K_CANDS
                k_distribution[K_batch] = (k_distribution.get(K_batch, 0)
                                             + cur_bs)
                conts, logps, temps, gflags, iflags = \
                    generate_k_candidates(
                        model, tokenizer, pids, pmask, K=K_batch,
                        max_new_tokens=max_new_tokens, pad_id=pad_id)
                buggy_list = batch['buggy_text']
                decoded = []
                for j in range(cur_bs):
                    cands = conts[j]
                    buggy = buggy_list[j].strip()[:2000]
                    cand_feats = extract_features(
                        buggy, cands, logps[j], temps[j],
                        gflags[j], iflags[j])
                    scores = reranker.predict(cand_feats)
                    greedy_idx = next((i for i, g in enumerate(gflags[j])
                                        if g), 0)
                    max_score = float(np.max(scores))
                    # Configurable margin — v12 default 0.005 (down from
                    # 0.015) now that reranker is trained with LambdaRank
                    # on 1600 candidates and empirically picks better
                    # when allowed to fire more aggressively.
                    if (max_score - scores[greedy_idx]) < RERANK_MARGIN:
                        best = greedy_idx
                    else:
                        best = int(np.argmax(scores))
                    decoded.append(cands[best])
                    all_greedy.append(cands[greedy_idx])

            for j, gt in enumerate(decoded):
                rt = batch['fixed_text'][j].strip()[:2000]
                all_gen.append(gt)
                all_ref.append(rt)
                if gt == rt:
                    em += 1
                total += 1

    cb = _compute_codebleu(all_gen, all_ref, lang=lang)
    cb_greedy = None
    if method == 'synthfix' and all_greedy:
        cb_greedy = _compute_codebleu(all_greedy, all_ref, lang=lang)
        P(f'  [{tag}] greedy-only CodeBLEU={cb_greedy*100:.2f}%  '
          f'reranked CodeBLEU={cb*100:.2f}%  '
          f'delta={(cb-cb_greedy)*100:+.2f}pp')

    results = {
        'method': method.upper() if method != 'synthfix' else 'SynthFix',
        'model': model_name,
        'dataset': dataset_name,
        'language': lang,
        'exact_match': em / total if total else 0,
        'codebleu': cb,
        'total': total,
        'em_count': em,
        'train_time_s': train_time,
        'gpu': gpu,
        'peak_gpu_mb': torch.cuda.max_memory_allocated(device) / 1e6,
        'config': {
            'batch_size': bs, 'epochs': epochs,
            'lora_rank': lora_rank, 'lr': lr,
            'max_new_tokens': max_new_tokens,
            'grad_checkpoint': args.grad_checkpoint,
            'seed': seed,
            'init_from_ckpt': getattr(args, 'init_from_ckpt', None),
            'rl_beta': getattr(args, 'rl_beta', None),
            'rloo_k': getattr(args, 'rloo_k', None),
            'rl_temp': getattr(args, 'rl_temp', None),
            'rl_top_p': getattr(args, 'rl_top_p', None),
            'rl_no_repeat_ngram': getattr(args, 'rl_no_repeat_ngram', None),
            'rerank_margin': RERANK_MARGIN,
            'num_rerank_cands': K_CANDS,
            'no_rerank': NO_RERANK,
            'rft_rl_temp': getattr(args, 'rft_rl_temp', None),
            'rft_no_repeat_ngram': getattr(args, 'rft_no_repeat_ngram', None),
        },
    }
    if method == 'synthfix':
        results['k_distribution'] = k_distribution
        results['reranker'] = (
            'lightgbm' if (reranker is not None and reranker.gbm is not None)
            else 'rule-based')
        if cb_greedy is not None:
            results['codebleu_greedy'] = cb_greedy
            results['codebleu_rerank'] = cb
            results['reranker_delta_pp'] = (cb - cb_greedy) * 100

    P(f'  [{tag}] EM={em}/{total}  CodeBLEU={cb*100:.2f}%  '
      f'time={train_time:.0f}s  peak={results["peak_gpu_mb"]:.0f}MB')

    Path(args.out).write_text(json.dumps(results, indent=2))
    P(f'  [{tag}] Saved → {args.out}')


# ── Single Experiment (one model × one dataset) ────────────────────────

# Shared command builder used by both the per-experiment helper and the
# global GPU-pool scheduler.  Kept at module scope so the scheduler can
# call it without a closure.
def _build_worker_cmd(model_cfg, dataset_name, data_dir, method, gpu, out):
    model = model_cfg['model']
    py = sys.executable
    script = str(Path(__file__).resolve())
    cmd = [
        py, '-u', script, '--worker',
        '--method', method,
        '--gpu', str(gpu),
        '--data_dir', data_dir,
        '--out', out,
        '--model_name', model,
        '--dataset_name', dataset_name,
        '--batch_size', str(model_cfg['batch_size']),
        '--epochs', str(model_cfg['epochs']),
        '--lora_rank', str(model_cfg['lora_rank']),
        '--lr', str(model_cfg['lr']),
        '--max_new_tokens', str(model_cfg['max_new_tokens']),
    ]
    if model_cfg['grad_checkpoint']:
        cmd.append('--grad_checkpoint')
    return cmd


def _collect_experiment_results(model_cfg, dataset_name, exp_dir, exp_time):
    """Gather SFT/RFT/SynthFix JSON files into a combined report."""
    model = model_cfg['model']
    tag = f'{model}/{dataset_name}'
    rfiles = {
        'SFT': exp_dir / 'sft.json',
        'RFT': exp_dir / 'rft.json',
        'SynthFix': exp_dir / 'synthfix.json',
    }
    results = {}
    for name, fp in rfiles.items():
        if fp.exists():
            results[name] = json.loads(fp.read_text())
        else:
            _log(f'  WARNING: {name} results missing for {tag}')
            results[name] = {'codebleu': 0, 'exact_match': 0,
                             'train_time_s': 0}

    sf_cb = results['SynthFix']['codebleu']
    best_bl = max(results['SFT']['codebleu'], results['RFT']['codebleu'])
    delta = sf_cb - best_bl
    _log(f'  {tag}: SFT={results["SFT"]["codebleu"]*100:.1f}%  '
         f'RFT={results["RFT"]["codebleu"]*100:.1f}%  '
         f'SF={sf_cb*100:.1f}%  delta={delta*100:+.1f}pp  '
         f'[{exp_time/60:.0f}min]')

    combined = {'model': model, 'dataset': dataset_name,
                'results': results, 'time_s': exp_time}
    (exp_dir / 'combined.json').write_text(json.dumps(combined, indent=2))
    return combined


def run_experiment(model_cfg, dataset_name, data_dir, exp_dir):
    """Legacy single-experiment runner (kept for backwards compat).

    Use ``run_suite_across_gpus`` for full-suite execution — it packs
    independent (method, model, dataset) jobs onto all available GPUs
    instead of leaving GPU 1 idle when baselines are cached.
    """
    model = model_cfg['model']
    tag = f'{model}/{dataset_name}'
    _log(f'{"─"*60}')
    _log(f'  {tag}  BS={model_cfg["batch_size"]}  ep={model_cfg["epochs"]}  '
         f'LoRA r={model_cfg["lora_rank"]}  lr={model_cfg["lr"]}')

    exp_dir.mkdir(parents=True, exist_ok=True)
    rfiles = {
        'SFT': str(exp_dir / 'sft.json'),
        'RFT': str(exp_dir / 'rft.json'),
        'SynthFix': str(exp_dir / 'synthfix.json'),
    }

    t0 = time.time()

    sft_cached = Path(rfiles['SFT']).exists()
    rft_cached = Path(rfiles['RFT']).exists()

    if sft_cached and rft_cached:
        _log(f'  SFT & RFT results cached — skipping baselines')
    else:
        procs = []
        if not sft_cached:
            procs.append(('SFT', subprocess.Popen(
                _build_worker_cmd(model_cfg, dataset_name, data_dir,
                                   'sft', 0, rfiles['SFT']),
                stdout=sys.stdout, stderr=sys.stderr)))
        else:
            _log(f'  SFT results cached — skipping')
        if not rft_cached:
            procs.append(('RFT', subprocess.Popen(
                _build_worker_cmd(model_cfg, dataset_name, data_dir,
                                   'rft', 1 if not sft_cached else 0,
                                   rfiles['RFT']),
                stdout=sys.stdout, stderr=sys.stderr)))
        else:
            _log(f'  RFT results cached — skipping')
        for name, p in procs:
            p.wait()
            _log(f'  {name} exit={p.returncode}')

    p_sf = subprocess.Popen(_build_worker_cmd(
        model_cfg, dataset_name, data_dir,
        'synthfix', 0, rfiles['SynthFix']),
        stdout=sys.stdout, stderr=sys.stderr)
    p_sf.wait()
    _log(f'  SynthFix exit={p_sf.returncode}')

    exp_time = time.time() - t0
    return _collect_experiment_results(model_cfg, dataset_name,
                                        exp_dir, exp_time)


# ── Full-suite scheduler ────────────────────────────────────────────────
#
# Packs all (method, model, dataset) jobs onto the set of available
# GPUs with a simple work-stealing pool: each GPU runs one job at a
# time; as soon as it finishes, the scheduler hands it the next queued
# job.  This keeps every GPU busy for the duration of the suite — much
# better than the per-experiment dispatcher which leaves GPU 1 idle
# whenever baselines are cached.
def run_suite_across_gpus(models, datasets, data_dirs, results_dir,
                           gpu_ids=(0, 1)):
    """Pack all independent training jobs across the listed GPUs.

    Ordering: SFT and RFT baselines first (they must exist before
    SynthFix's reranker training can meaningfully be compared), then
    SynthFix for every (model, dataset).  Already-cached results are
    skipped so the function is idempotent.
    """
    t0 = time.time()
    job_queue = []  # list of (tag, method, model_cfg, ds_name, out_path)

    for method in ('sft', 'rft', 'synthfix'):
        for model_cfg in models:
            for ds_name in datasets:
                exp_dir = results_dir / model_cfg['model'] / ds_name
                exp_dir.mkdir(parents=True, exist_ok=True)
                fn = {'sft': 'sft.json', 'rft': 'rft.json',
                      'synthfix': 'synthfix.json'}[method]
                out = exp_dir / fn
                # SynthFix always re-runs (we iterate on it); baselines
                # are cached and skipped if present.
                if out.exists() and method != 'synthfix':
                    _log(f'  [cache] {method.upper()} {model_cfg["model"]}/'
                         f'{ds_name} — skipping')
                    continue
                tag = f'{method}/{model_cfg["model"]}/{ds_name}'
                job_queue.append((tag, method, model_cfg, ds_name, str(out)))

    _log(f'Scheduler: {len(job_queue)} jobs across '
         f'{len(gpu_ids)} GPUs ({gpu_ids})')

    running = {gid: None for gid in gpu_ids}   # gpu_id -> (tag, Popen)
    completed = []

    def _launch(gid, job):
        tag, method, model_cfg, ds_name, out = job
        cmd = _build_worker_cmd(model_cfg, ds_name,
                                 data_dirs[ds_name], method, gid, out)
        _log(f'  [GPU{gid} ←] starting {tag}')
        p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
        running[gid] = (tag, p)

    # Prime GPUs with initial jobs
    for gid in gpu_ids:
        if job_queue:
            _launch(gid, job_queue.pop(0))

    while any(v is not None for v in running.values()):
        for gid in gpu_ids:
            if running[gid] is None:
                continue
            tag, p = running[gid]
            rc = p.poll()
            if rc is None:
                continue
            _log(f'  [GPU{gid} ✓] {tag} exit={rc}  '
                 f'({(time.time()-t0)/60:.0f}min elapsed)')
            completed.append((tag, rc))
            running[gid] = None
            if job_queue:
                _launch(gid, job_queue.pop(0))
        time.sleep(2)

    # Aggregate results per (model, dataset)
    all_results = {}
    for model_cfg in models:
        for ds_name in datasets:
            exp_dir = results_dir / model_cfg['model'] / ds_name
            combined = _collect_experiment_results(
                model_cfg, ds_name, exp_dir, time.time() - t0)
            all_results[f'{model_cfg["model"]}/{ds_name}'] = combined

    _log(f'\nAll {len(completed)} jobs done in '
         f'{(time.time()-t0)/60:.1f} min')
    return all_results


# ── Orchestrator ────────────────────────────────────────────────────────

def main():
    global _log_file

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(RESULTS_DIR / 'experiment_log.txt', 'w')

    total_jobs = len(MODELS) * len(DATASETS)
    _log('=' * 70)
    _log('  SynthFix: FULL EXPERIMENT SUITE (rebuttal-matched config)')
    _log(f'  {len(MODELS)} models × {len(DATASETS)} datasets = {total_jobs} experiments')
    _log(f'  Datasets: {", ".join(DATASETS.keys())}')
    _log(f'  Models: {", ".join(m["model"] for m in MODELS)}')
    _log(f'  Config: lr=2e-4  epochs=4  LoRA r=16  alpha=32')
    _log('=' * 70)

    t_global = time.time()

    _log('Preparing all datasets...')
    data_dirs = prepare_all_data()
    _log(f'Data ready: {data_dirs}')

    # Detect visible GPUs and pack jobs across all of them using the
    # work-stealing scheduler (keeps every GPU busy for the full suite).
    try:
        n_gpu = torch.cuda.device_count()
    except Exception:
        n_gpu = 1
    gpu_ids = tuple(range(max(1, n_gpu)))
    _log(f'Scheduling across GPUs {gpu_ids}')

    all_results = run_suite_across_gpus(
        models=MODELS, datasets=DATASETS, data_dirs=data_dirs,
        results_dir=RESULTS_DIR, gpu_ids=gpu_ids)

    total_time = time.time() - t_global

    # ── Final Summary ───────────────────────────────────────────────
    _log('\n' + '=' * 70)
    _log('  FINAL SUMMARY — ALL EXPERIMENTS')
    _log('=' * 70)

    for ds_name in DATASETS:
        _log(f'\n  Dataset: {ds_name} ({DATASETS[ds_name]["language"]})')
        _log(f'  {"Model":<20} {"SFT":>8} {"RFT":>8} {"SynthFix":>10} '
             f'{"Delta":>8} {"Win":>4}')
        _log(f'  {"-"*60}')
        for model_cfg in MODELS:
            key = f'{model_cfg["model"]}/{ds_name}'
            d = all_results.get(key, {})
            if 'error' in d:
                _log(f'  {model_cfg["model"]:<20} ERROR')
                continue
            r = d.get('results', {})
            sft = r.get('SFT', {}).get('codebleu', 0) * 100
            rft = r.get('RFT', {}).get('codebleu', 0) * 100
            sf = r.get('SynthFix', {}).get('codebleu', 0) * 100
            best = max(sft, rft)
            delta = sf - best
            win = 'Y' if delta > 0 else '-'
            _log(f'  {model_cfg["model"]:<20} {sft:>7.1f}% {rft:>7.1f}% '
                 f'{sf:>9.1f}% {delta:>+7.1f} {win:>4}')

    _log(f'\n  Total wall time: {total_time/3600:.1f} hours')
    _log('=' * 70)

    master = {
        'timestamp': datetime.now().isoformat(),
        'total_time_s': total_time,
        'datasets': {k: v for k, v in DATASETS.items()},
        'models': [m['model'] for m in MODELS],
        'experiments': all_results,
    }
    master_path = RESULTS_DIR / 'all_results.json'
    master_path.write_text(json.dumps(master, indent=2, default=str))
    _log(f'Master results: {master_path}')

    if _log_file:
        _log_file.close()


# ── CLI ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--method', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--out', type=str)
    parser.add_argument('--model_name', type=str, default='deepseek-1.3b')
    parser.add_argument('--dataset_name', type=str, default='fixjs')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--grad_checkpoint', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--init_from_ckpt', type=str, default=None,
                        help='Path to a LoRA ckpt to initialise from. '
                             'Used to bootstrap RFT/SynthFix from SFT.')
    parser.add_argument('--save_ckpt_to', type=str, default=None,
                        help='Save the best LoRA ckpt to this path '
                             '(two-stage training).')
    parser.add_argument('--sft_warmup_epochs', type=int, default=-1,
                        help='Override SFT warmup epochs for SynthFix '
                             '(default = module default; 0 when starting '
                             'from a trained SFT ckpt).')
    # ── v12 deadline-run knobs ──────────────────────────────────────
    parser.add_argument('--rl_beta', type=float, default=None,
                        help='Override SynthFix BETA_RL (RL loss weight).')
    parser.add_argument('--rloo_k', type=int, default=None,
                        help='Override SynthFix RLOO_K (rollouts/prompt).')
    parser.add_argument('--rl_temp', type=float, default=None,
                        help='Override SynthFix RL_SAMPLE_TEMP.')
    parser.add_argument('--rl_top_p', type=float, default=None,
                        help='Override SynthFix RL sampling top-p.')
    parser.add_argument('--rl_no_repeat_ngram', type=int, default=None,
                        help='Override SynthFix RL no_repeat_ngram_size.')
    parser.add_argument('--rft_rl_temp', type=float, default=0.8,
                        help='Temperature for RFT REINFORCE rollouts.')
    parser.add_argument('--rft_no_repeat_ngram', type=int, default=0,
                        help='no_repeat_ngram_size for RFT rollouts (0=off).')
    parser.add_argument('--rerank_margin', type=float, default=0.015,
                        help='Score-margin threshold before the reranker '
                             'is allowed to override greedy at test time.')
    parser.add_argument('--num_rerank_cands', type=int, default=16,
                        help='K value for candidate generation at test time '
                             '(and at reranker-training time).')
    parser.add_argument('--no_rerank', action='store_true',
                        help='Disable the learned reranker; use greedy '
                             'decode at test time (SynthFix ablation).')
    args = parser.parse_args()

    if args.worker:
        run_worker(args)
    else:
        main()
