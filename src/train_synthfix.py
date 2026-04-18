"""
SynthFix — Router-gated REINFORCE with split-symbolic reward

Training-time pipeline (paired with the inference-time best-of-K
reranker in run_all_experiments.py).  A small router network decides,
per sample, how much weight the symbolic reward signal should carry
at training time:

  Epoch 1           Pure SFT warmup.  Gives the policy a sane starting
                    distribution (random generations would make the RL
                    signal pure noise) and records per-sample SFT loss,
                    which the router uses as one of its features.
  Router pretrain   Supervised step: predict P(above-median loss).
  Epoch 2..E        For each batch:
                      (a) SFT CE on ground truth (anchor, always on).
                      (b) Generate K=1 continuation per sample (sampled).
                      (c) Compute split-symbolic reward
                            r = w_ast·r_AST + w_cfg·r_CFG
                              + w_sem·r_SEM + w_sim·r_SIM
                      (d) Normalized advantage = (r - baseline) / std
                      (e) REINFORCE loss per sample =
                            advantage × CE_on_sampled
                      (f) Router-gated RL: multiply each sample's RL
                          loss by prob_hard (continuous gate) — easy
                          samples → gate≈0 → only SFT; hard samples →
                          gate≈1 → SFT + reward guidance.
                      (g) Total loss = sft_loss + beta × router_RL_loss.
                    In parallel the router is trained (BCE) to predict
                    "above-median loss" — i.e. its idea of "hard".

Selection: best val_loss, same as SFT / RFT baselines.

Design notes:
  * Unfiltered RFT spends the reward signal on easy samples that SFT
    already nails, which is pure variance.  Router gating focuses the
    noisy signal on hard samples, where it is most useful.
  * The split-symbolic reward is a richer, less noisy training signal
    than a single scalar (SIM alone can dominate if target ≈ buggy).
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import types as _types
_sys = sys

if 'torch._dynamo' not in _sys.modules:
    _m = _types.ModuleType('torch._dynamo')
    _m.disable = lambda fn=None, recursive=True: fn if fn else (lambda f: f)
    _m.graph_break = lambda: None
    _m.is_compiling = lambda: False
    _mc = _types.ModuleType('torch._dynamo.config')
    _mc.suppress_errors = True
    _m.config = _mc
    _sys.modules['torch._dynamo'] = _m
    _sys.modules['torch._dynamo.config'] = _mc

import math
import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.backends.cudnn.benchmark = True

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from data.dataset import RepairDataset, collate_fn, create_dataloaders
from models.reward import compute_reward
from models.router import RouterModel, compute_batch_features, normalize_features
from models.symbolic import (classify_token_string, compute_reward_split,
                              compute_reward_from_split)

# ── Model registry ──────────────────────────────────────────────────────

MODEL_PATHS = {
    'deepseek-1.3b': 'deepseek-ai/deepseek-coder-1.3b-base',
    'starcoder2-3b': 'bigcode/starcoder2-3b',
    'codellama-7b': 'codellama/CodeLlama-7b-hf',
    'deepseek-6.7b': 'deepseek-ai/deepseek-coder-6.7b-base',
    'starcoder2-7b': 'bigcode/starcoder2-7b',
}

LORA_TARGET_MODULES = {
    'deepseek-1.3b': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    'starcoder2-3b': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    'codellama-7b': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    'deepseek-6.7b': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    'starcoder2-7b': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
}


def get_lora_config(model_name: str, lora_rank: int = 16) -> LoraConfig:
    targets = LORA_TARGET_MODULES.get(model_name,
                                       ['q_proj', 'v_proj', 'k_proj', 'o_proj'])
    return LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=targets,
        lora_dropout=0.05,
        bias='none',
        task_type=TaskType.CAUSAL_LM,
    )


def _log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Metrics ─────────────────────────────────────────────────────────────

def _detect_language(dataset_path: str) -> str:
    p = str(dataset_path).lower()
    if 'codeflaws' in p or '/c/' in p:
        return 'c'
    if 'sven' in p or '/python/' in p:
        return 'python'
    return 'javascript'


def _compute_codebleu(preds, refs, lang='javascript'):
    try:
        from codebleu import calc_codebleu
        MAX_CHARS = 2000
        preds_t = [p[:MAX_CHARS] for p in preds]
        refs_t = [r[:MAX_CHARS] for r in refs]
        result = calc_codebleu(
            references=[[r] for r in refs_t],
            predictions=preds_t,
            lang=lang, weights=(0.25, 0.25, 0.25, 0.25),
        )
        return result['codebleu']
    except Exception as e:
        _log(f'CodeBLEU error: {e}')
        return 0.0


# ── Token type lookup table ────────────────────────────────────────────

def build_token_type_table(tokenizer) -> torch.Tensor:
    """Precompute symbolic type (0=AST, 1=CFG, 2=SEM, 3=SIM) per vocab id.

    Called once at startup. Returns a (vocab_size,) int tensor on CPU.
    Looking up tags during training is then a single advanced-index
    op — O(1) per token, no Python per-step.
    """
    vocab_size = len(tokenizer)
    tags = torch.full((vocab_size,), 3, dtype=torch.long)  # default SIM
    for tid in range(vocab_size):
        try:
            s = tokenizer.decode([tid], skip_special_tokens=False)
        except Exception:
            continue
        tags[tid] = classify_token_string(s)
    return tags


# ── Per-token weighted loss ────────────────────────────────────────────

def _weighted_ce(logits, labels, token_type_table, type_weights):
    """Cross-entropy with per-token symbolic weights.

    The per-token weights are rescaled so that mean(weight) over
    non-masked tokens in the batch equals 1.0. This preserves the
    overall gradient magnitude — only the relative attention across
    token TYPES shifts. Crucially, SFT is recovered exactly by setting
    type_weights to [1, 1, 1, 1].

    Args:
      logits: (bs, seq_len, vocab)
      labels: (bs, seq_len)   target ids, -100 on prompt/pad
      token_type_table: (vocab,) long tensor on model device
      type_weights: (4,) float tensor on model device

    Returns:
      (per_sample_loss, weighted_token_loss) — both scalars for the
      second are aggregated over the batch.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    bs, sl_m1, vocab = shift_logits.shape

    per_token = F.cross_entropy(
        shift_logits.view(-1, vocab),
        shift_labels.view(-1),
        reduction='none', ignore_index=-100,
    ).view(bs, sl_m1)
    mask = (shift_labels != -100).float()

    # Look up token type for each label position. For masked (-100)
    # positions we fetch type 3 (SIM) by clamping; they're zeroed by
    # the mask anyway so the value doesn't matter.
    safe_labels = shift_labels.clamp(min=0)
    tok_types = token_type_table[safe_labels]  # (bs, sl_m1)
    tok_weights_raw = type_weights[tok_types]  # (bs, sl_m1)
    tok_weights = tok_weights_raw * mask

    # Magnitude-preserving normalization: so mean weight over non-masked
    # tokens equals 1.0. Overall gradient scale is identical to plain CE.
    denom = tok_weights.sum()
    n_active = mask.sum()
    if denom.item() > 0 and n_active.item() > 0:
        tok_weights = tok_weights * (n_active / denom)

    weighted_loss = (per_token * tok_weights).sum() / n_active.clamp(min=1)

    # Per-sample unweighted (for router curriculum and diagnostics)
    per_sample = (per_token * mask).sum(1) / mask.sum(1).clamp(min=1)

    return per_sample, weighted_loss


def _compute_per_sample_loss_plain(model, input_ids, attn_mask, labels):
    """SFT-identical loss: (per_sample, token_loss)."""
    outputs = model(input_ids=input_ids, attention_mask=attn_mask)
    logits = outputs.logits
    bs = input_ids.size(0)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction='none', ignore_index=-100,
    ).view(bs, -1)
    tok_mask = (shift_labels != -100).float()
    per_sample = (per_token * tok_mask).sum(1) / tok_mask.sum(1).clamp(min=1)
    token_loss = (per_token * tok_mask).sum() / tok_mask.sum().clamp(min=1)
    return per_sample, token_loss


# ── Core Training ───────────────────────────────────────────────────────

ROUTER_LR = 5e-4
ROUTER_PRETRAIN_STEPS = 100
VAL_GEN_MAX_SAMPLES = 64
VAL_GEN_MAX_NEW_TOKENS = 96

# Training-loop config
SFT_WARMUP_EPOCHS = 1
# RL loss weight.  With RLOO (k samples per prompt, leave-one-out
# baseline) the advantage signal is much less noisy than with a
# single-sample moving-average baseline, so a moderate BETA_RL is
# safe and gives the reranker diverse-but-still-sharp candidates.
# Module-level default; overridable per-run via args.rl_beta.
BETA_RL = 0.25
# Number of on-policy samples per prompt for RLOO.  k=2 is the
# minimal setting (one sample acts as the baseline for the other)
# and already removes most of the REINFORCE variance.
RLOO_K = 2
# Temperature for REINFORCE continuation sampling.  0.9 keeps the
# sampled candidates meaningfully different from greedy decoding so
# advantages stay informative on small models / small datasets.
RL_SAMPLE_TEMP = 0.9
RL_TOP_P = 0.95
RL_NO_REPEAT_NGRAM = 3
RL_MAX_GEN = 128
# Reward mixture weights.  Heavy on GT-aligned components (CFG-match
# and chrF similarity to target) so that REINFORCE pushes the policy
# toward GT-like continuations rather than arbitrary high-AST/high-
# SEM surface forms — the latter was observed to drift greedy test
# generations away from the reference on small datasets.
REWARD_WEIGHTS = dict(lambda_ast=0.10, lambda_cfg=0.30,
                       lambda_sem=0.10, lambda_sim=0.50)


def _pretrain_router(router, router_optimizer, train_loader, loss_history,
                     device, steps=ROUTER_PRETRAIN_STEPS):
    """Supervised pre-training: predict which samples have above-median loss."""
    all_losses = list(loss_history.values())
    median_loss = sorted(all_losses)[len(all_losses) // 2]
    _log(f'  Router pre-training: {steps} steps, '
         f'median_loss={median_loss:.4f}, '
         f'{sum(1 for v in all_losses if v > median_loss)}/{len(all_losses)} above median')

    router.train()
    step = 0
    last_loss = torch.tensor(0.0)
    for batch in train_loader:
        if step >= steps:
            break
        buggy_texts = batch['buggy_text']
        sample_indices = batch['sample_indices']
        features = compute_batch_features(buggy_texts, sample_indices,
                                          loss_history)
        features_norm = normalize_features(features).to(device)
        pred = router(features_norm).squeeze(-1)
        targets = torch.tensor(
            [1.0 if loss_history.get(idx, 0) > median_loss else 0.0
             for idx in sample_indices], device=device)
        loss = F.binary_cross_entropy(pred, targets)
        router_optimizer.zero_grad()
        loss.backward()
        router_optimizer.step()
        last_loss = loss
        step += 1

    _log(f'  Router pre-training done ({step} steps, '
         f'final_loss={last_loss.item():.4f})')


def _compute_val_loss(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            ids = batch['input_ids'].to(device, non_blocking=True)
            mask = batch['attention_mask'].to(device, non_blocking=True)
            lab = batch['labels'].to(device, non_blocking=True)
            out = model(input_ids=ids, attention_mask=mask, labels=lab)
            total_loss += out.loss.item()
            n += 1
    model.train()
    return total_loss / max(n, 1)


def _compute_val_codebleu(model, tokenizer, val_loader, device, lang,
                           max_samples=VAL_GEN_MAX_SAMPLES,
                           max_new_tokens=VAL_GEN_MAX_NEW_TOKENS):
    """Greedy-decode on val subset → CodeBLEU. Model-selection signal."""
    model.eval()
    preds, refs = [], []
    with torch.no_grad():
        for batch in val_loader:
            if len(preds) >= max_samples:
                break
            pids = batch['prompt_input_ids'].to(device, non_blocking=True)
            pmask = batch['prompt_attention_mask'].to(device, non_blocking=True)
            gen = model.generate(
                input_ids=pids, attention_mask=pmask,
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id)
            prompt_pad_len = pids.size(1)
            for j, g in enumerate(gen):
                if len(preds) >= max_samples:
                    break
                pred = tokenizer.decode(
                    g[prompt_pad_len:], skip_special_tokens=True
                ).strip()[:2000]
                ref = batch['fixed_text'][j].strip()[:2000]
                preds.append(pred)
                refs.append(ref)
    model.train()
    return _compute_codebleu(preds, refs, lang=lang)


def train_synthfix(model, tokenizer, router, train_loader, val_loader,
                   output_dir, args, device):
    """SynthFix — split-symbolic curriculum training."""
    model = model.to(device)
    router = router.to(device)

    # Per-run overrides of module-level RL config.  Default to module
    # values so existing callers (train_synthfix CLI) are unaffected.
    beta_rl    = float(getattr(args, 'rl_beta',  BETA_RL))
    rloo_k     = int(  getattr(args, 'rloo_k',   RLOO_K))
    rl_temp    = float(getattr(args, 'rl_temp',  RL_SAMPLE_TEMP))
    rl_top_p   = float(getattr(args, 'rl_top_p', RL_TOP_P))
    rl_no_rep  = int(  getattr(args, 'rl_no_repeat_ngram', RL_NO_REPEAT_NGRAM))

    model_optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                         weight_decay=0.01)
    router_optimizer = torch.optim.Adam(router.parameters(), lr=ROUTER_LR)

    # Cosine schedule over total steps
    total_steps = max(1, len(train_loader) * args.epochs)

    def lr_at_step(step):
        # Linear warmup over 5% of total steps, cosine decay thereafter
        warmup = max(1, int(0.05 * total_steps))
        if step < warmup:
            return args.lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    lang = _detect_language(getattr(args, 'dataset', ''))

    # Token-type table kept around so the inference reranker (run via
    # run_all_experiments.py) can reuse it for split-symbolic features
    # on generated candidates.  Not used in the training loss anymore
    # (Per-token CE weighting is not used in the current training
    # loss; split-symbolic lives in the reward function and at
    # inference time instead.)
    _log('Building token type table (for inference reranker)...')
    token_type_table = build_token_type_table(tokenizer).to(device)
    _log(f'Token type table: total={len(token_type_table)}  '
         f'AST={(token_type_table == 0).sum().item()}  '
         f'CFG={(token_type_table == 1).sum().item()}  '
         f'SEM={(token_type_table == 2).sum().item()}  '
         f'SIM={(token_type_table == 3).sum().item()}')

    loss_history = {}
    best_val_cb = -1.0
    best_val_loss = float('inf')
    best_epoch = -1
    best_model_dir = None

    total_sft_steps = 0

    # Allow a caller (e.g. the two-stage orchestrator) to skip the
    # SFT warmup when starting from an already-SFT-trained checkpoint.
    sft_warmup = getattr(args, 'sft_warmup_epochs', SFT_WARMUP_EPOCHS)
    warmup_epochs = min(sft_warmup, max(0, args.epochs - 1)) if sft_warmup > 0 else 0

    _log('=' * 70)
    _log('SYNTHFIX: Router-gated REINFORCE + split-symbolic reward')
    _log(f'  Phase 1 (epoch 1..{warmup_epochs}): pure SFT warmup')
    _log(f'  Router pre-training: {ROUTER_PRETRAIN_STEPS} supervised steps')
    _log(f'  Phase 2 (epochs {warmup_epochs+1}..{args.epochs}): SFT + gated RFT')
    _log(f'    loss = sft_ce + {beta_rl} * E[hard_gate * adv_LOO * CE_sampled]  '
         f'(RLOO K={rloo_k})')
    _log(f'    reward = {REWARD_WEIGHTS["lambda_ast"]}*r_AST + '
         f'{REWARD_WEIGHTS["lambda_cfg"]}*r_CFG + '
         f'{REWARD_WEIGHTS["lambda_sem"]}*r_SEM + '
         f'{REWARD_WEIGHTS["lambda_sim"]}*r_SIM')
    _log(f'    sample T={rl_temp} top_p={rl_top_p} '
         f'no_repeat_ngram={rl_no_rep}  max_new_tokens={RL_MAX_GEN}')
    _log(f'  Selection: best val loss')
    _log(f'  Batches/epoch: {len(train_loader)}  LR={args.lr}  '
         f'cosine schedule  lang={lang}')
    _log('=' * 70)

    pad_id = tokenizer.pad_token_id
    rl_baseline = 0.0
    rl_baseline_mom = 0.9

    # When the caller skips SFT warmup (two-stage: start from an
    # already-SFT-trained ckpt), we still need to populate loss_history
    # and pretrain the router. Do a single no-grad pass over the train
    # set and then pretrain the router.
    if warmup_epochs == 0 and len(loss_history) == 0:
        _log('  [init] No warmup: priming loss_history with one no-grad pass')
        model.eval()
        with torch.no_grad():
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attn_mask = batch['attention_mask'].to(device,
                                                        non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)
                sample_indices = batch['sample_indices']
                per_sample, _ = _compute_per_sample_loss_plain(
                    model, input_ids, attn_mask, labels)
                for i in range(input_ids.size(0)):
                    loss_history[sample_indices[i]] = per_sample[i].item()
        _log(f'  [init] loss_history size: {len(loss_history)}')
        _pretrain_router(router, router_optimizer, train_loader,
                         loss_history, device)
        model.train()

    for epoch in range(args.epochs):
        model.train()
        router.train()
        is_curriculum = epoch >= warmup_epochs and len(loss_history) > 0

        if epoch == warmup_epochs and len(loss_history) > 0:
            _pretrain_router(router, router_optimizer, train_loader,
                             loss_history, device)

        epoch_loss = 0.0
        epoch_rl = 0.0
        epoch_reward = 0.0
        n_batches = 0
        epoch_hard = 0

        phase_tag = 'RFT+GATE' if is_curriculum else 'SFT'
        progress = tqdm(train_loader,
                        desc=f'Epoch {epoch+1}/{args.epochs} [{phase_tag}]')

        for batch in progress:
            # Apply cosine LR
            for g in model_optimizer.param_groups:
                g['lr'] = lr_at_step(total_sft_steps)

            buggy_texts = batch['buggy_text']
            sample_indices = batch['sample_indices']
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attn_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            bs = input_ids.size(0)

            if not is_curriculum:
                # Phase-1 SFT: single forward-backward AND record per-sample
                # loss for the router features / history.
                per_sample, sft_loss = _compute_per_sample_loss_plain(
                    model, input_ids, attn_mask, labels)
                with torch.no_grad():
                    for i in range(bs):
                        loss_history[sample_indices[i]] = per_sample[i].item()
                model_optimizer.zero_grad()
                sft_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                model_optimizer.step()
                epoch_loss += sft_loss.item()
                total_sft_steps += 1
                n_batches += 1
                progress.set_postfix(
                    loss=f'{epoch_loss / n_batches:.4f}',
                    phase=phase_tag,
                    lr=f'{model_optimizer.param_groups[0]["lr"]:.2e}',
                )
                continue
            else:
                # Phase-2 curriculum: record per_sample under no_grad to
                # update loss_history without keeping a graph in memory.
                with torch.no_grad():
                    per_sample, _ = _compute_per_sample_loss_plain(
                        model, input_ids, attn_mask, labels)
                    for i in range(bs):
                        loss_history[sample_indices[i]] = per_sample[i].item()
                # (b) RLOO: sample K=2 continuations per prompt and use
                # each one as the baseline for the other.  This is an
                # unbiased, variance-reduced REINFORCE estimator that
                # avoids the drift seen with a single-sample + moving-
                # average baseline on small datasets.
                prompt_ids = batch['prompt_input_ids'].to(device,
                                                           non_blocking=True)
                prompt_mask = batch['prompt_attention_mask'].to(
                    device, non_blocking=True)
                prompt_len = prompt_ids.size(1)
                fixed_texts = batch['fixed_text']

                gen_ids_list = []
                rewards_list = []
                for _ in range(rloo_k):
                    with torch.no_grad():
                        g = model.generate(
                            input_ids=prompt_ids,
                            attention_mask=prompt_mask,
                            max_new_tokens=RL_MAX_GEN, do_sample=True,
                            temperature=rl_temp, top_p=rl_top_p,
                            no_repeat_ngram_size=rl_no_rep,
                            pad_token_id=pad_id,
                        )
                    rs_list = []
                    for j in range(g.size(0)):
                        cont = tokenizer.decode(
                            g[j, prompt_len:], skip_special_tokens=True)
                        rs = compute_reward_split(cont, fixed_texts[j])
                        r = compute_reward_from_split(rs, **REWARD_WEIGHTS)
                        rs_list.append(float(r))
                    gen_ids_list.append(g)
                    rewards_list.append(torch.tensor(
                        rs_list, device=device, dtype=torch.float32))

                # (c) RLOO advantages: baseline for sample k = mean of
                # the other K-1 samples.  For K=2 this is just the
                # other sample's reward.
                R = torch.stack(rewards_list, dim=0)          # (K, bs)
                sum_R = R.sum(dim=0, keepdim=True)            # (1, bs)
                loo_baseline = (sum_R - R) / max(rloo_k - 1, 1)
                advantages_all = R - loo_baseline             # (K, bs)
                adv_std = advantages_all.std().clamp(min=1e-4)
                advantages_all = advantages_all / adv_std
                # Positive-advantage-only: never push away from a
                # sample, only amplify samples that beat their LOO
                # baseline.
                advantages_all = advantages_all.clamp(min=0.0)

                mean_reward = R.mean().item()
                # Keep moving baseline just for logging.
                rl_baseline = (rl_baseline_mom * rl_baseline
                               + (1.0 - rl_baseline_mom) * mean_reward)

                # (d) Router gate — hard-only on truly hard samples.
                features = compute_batch_features(
                    buggy_texts, sample_indices, loss_history)
                features_norm = normalize_features(features).to(device)
                with torch.no_grad():
                    prob_hard = router(features_norm).squeeze(-1)
                    hard_gate = (prob_hard >= 0.5).float()
                epoch_hard += int(hard_gate.sum())

                # (e) Per-sample CE per candidate, backward separately
                # so only one forward graph lives in memory at a time.
                model_optimizer.zero_grad()
                rl_loss_val = 0.0

                def _rl_backward(gen_ids_k, adv_k):
                    full_mask_k = (gen_ids_k != pad_id).long()
                    labels_k = gen_ids_k.clone()
                    labels_k[:, :prompt_len] = -100
                    labels_k[full_mask_k == 0] = -100
                    out_k = model(input_ids=gen_ids_k,
                                  attention_mask=full_mask_k)
                    shift_logits = out_k.logits[..., :-1, :].contiguous()
                    shift_labels = labels_k[..., 1:].contiguous()
                    bs_, sl_m1, vocab = shift_logits.shape
                    per_tok = F.cross_entropy(
                        shift_logits.view(-1, vocab),
                        shift_labels.view(-1),
                        reduction='none', ignore_index=-100,
                    ).view(bs_, sl_m1)
                    tok_mask = (shift_labels != -100).float()
                    n_tok = tok_mask.sum(1)
                    valid_k = (n_tok > 0).float()
                    per_sample_ce_k = (per_tok * tok_mask).sum(1) \
                                        / n_tok.clamp(min=1)
                    rl_per = adv_k * per_sample_ce_k
                    # Divide by rloo_k so the gradient magnitude is
                    # comparable to the single-sample loss.
                    rl_loss_k = (hard_gate * rl_per * valid_k).sum() \
                        / (hard_gate * valid_k).sum().clamp(min=1e-3) \
                        / float(rloo_k)
                    (beta_rl * rl_loss_k).backward()
                    val_k = rl_loss_k.item()
                    del out_k, shift_logits, shift_labels, per_tok
                    del per_sample_ce_k, rl_loss_k
                    return val_k

                for k in range(rloo_k):
                    rl_loss_val += _rl_backward(gen_ids_list[k],
                                                  advantages_all[k])

                epoch_rl += rl_loss_val
                epoch_reward += mean_reward

                # Router supervised update (uses detached features, no
                # policy grad interference).
                ps_detached = per_sample.detach()
                targets_bin = (ps_detached > ps_detached.median()).float()
                prob_grad = router(features_norm).squeeze(-1)
                router_loss = F.binary_cross_entropy(prob_grad, targets_bin)
                router_optimizer.zero_grad()
                router_loss.backward()
                router_optimizer.step()

                # SFT anchor backward.
                sft_out2 = model(input_ids=input_ids,
                                  attention_mask=attn_mask, labels=labels)
                sft_out2.loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                model_optimizer.step()
                epoch_loss += sft_out2.loss.item()

            total_sft_steps += 1
            n_batches += 1
            progress.set_postfix(
                loss=f'{epoch_loss / n_batches:.4f}',
                rl=f'{epoch_rl / max(n_batches,1):.3f}',
                reward=f'{epoch_reward / max(n_batches,1):.3f}',
                phase=phase_tag,
                hard=epoch_hard,
                lr=f'{model_optimizer.param_groups[0]["lr"]:.2e}',
            )

        avg_loss = epoch_loss / max(n_batches, 1)
        val_loss = _compute_val_loss(model, val_loader, device)
        val_cb = _compute_val_codebleu(model, tokenizer, val_loader, device,
                                        lang)

        _log(f'Epoch {epoch+1}: train_loss={avg_loss:.4f}  '
             f'val_loss={val_loss:.4f}  '
             f'val_codebleu={val_cb*100:.2f}%  '
             f'phase={phase_tag}  hard={epoch_hard}  '
             f'loss_history_size={len(loss_history)}')

        ckpt = Path(output_dir) / f'checkpoint_epoch{epoch+1}'
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        torch.save(router.state_dict(), ckpt / 'router.pt')

        # Selection: best val_loss (matches SFT baseline criterion for
        # apples-to-apples). We ALSO track val_codebleu for logging but
        # don't use it as the tiebreaker.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_cb = val_cb
            best_epoch = epoch + 1
            best_model_dir = ckpt
            _log(f'  -> New best model (epoch {best_epoch}, '
                 f'val_loss={best_val_loss:.4f}, '
                 f'val_codebleu={best_val_cb*100:.2f}%)')

    _log(f'Training done: {total_sft_steps} steps, '
         f'best_epoch={best_epoch} (val_loss={best_val_loss:.4f}, '
         f'val_codebleu={best_val_cb*100:.2f}%)')

    # Reload best-epoch model if it wasn't the last one
    if best_model_dir is not None and best_epoch < args.epochs:
        _log(f'Reloading best model from epoch {best_epoch}')
        from peft import PeftModel
        model = model.base_model.model
        model = PeftModel.from_pretrained(model, str(best_model_dir))
    else:
        _log(f'Using final epoch model (best_epoch={best_epoch})')

    final = Path(output_dir) / 'final_model'
    final.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    torch.save(router.state_dict(), final / 'router.pt')
    return model, router


# ── Evaluation ──────────────────────────────────────────────────────────

def evaluate(model, tokenizer, test_loader, device, dataset_path=''):
    lang = _detect_language(dataset_path)
    model.eval()
    em = total = 0
    all_gen, all_ref = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            pids = batch['prompt_input_ids'].to(device, non_blocking=True)
            pmask = batch['prompt_attention_mask'].to(device, non_blocking=True)
            gen = model.generate(
                input_ids=pids, attention_mask=pmask,
                max_new_tokens=128, do_sample=False,
                pad_token_id=tokenizer.pad_token_id)
            prompt_pad_len = pids.size(1)
            for j, g in enumerate(gen):
                gt = tokenizer.decode(
                    g[prompt_pad_len:], skip_special_tokens=True
                ).strip()[:2000]
                rt = batch['fixed_text'][j].strip()[:2000]
                all_gen.append(gt)
                all_ref.append(rt)
                if gt == rt:
                    em += 1
                total += 1

    cb = _compute_codebleu(all_gen, all_ref, lang=lang)
    return {
        'exact_match': em / total if total else 0,
        'codebleu': cb,
        'total': total,
        'em_count': em,
    }


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='SynthFix training')
    parser.add_argument('--model', type=str, default='deepseek-1.3b',
                        choices=list(MODEL_PATHS.keys()))
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--lambda_ast', type=float, default=0.333)
    parser.add_argument('--lambda_cfg', type=float, default=0.333)
    parser.add_argument('--lambda_sem', type=float, default=0.334)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--grad_checkpoint', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available()
                          else 'cpu')
    _log(f'Device: {device}')

    model_path = MODEL_PATHS[args.model]
    _log(f'Loading {model_path}')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map={'': str(device)})
    model = get_peft_model(model, get_lora_config(args.model, args.lora_rank))

    if args.grad_checkpoint:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    _log(f'LoRA r={model.peft_config["default"].r}  '
         f'Params: {trainable:,} / {total_p:,} ({100*trainable/total_p:.2f}%)')

    router = RouterModel(input_size=4, hidden_size=64)

    train_loader, val_loader, test_loader = create_dataloaders(
        args.dataset, tokenizer, args.batch_size, args.max_length,
        num_workers=args.num_workers)

    model, router = train_synthfix(
        model, tokenizer, router, train_loader, val_loader,
        args.output, args, device)

    results = evaluate(model, tokenizer, test_loader, device, args.dataset)
    _log(f'Results: {json.dumps(results, indent=2)}')
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / 'results.json').write_text(
        json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
