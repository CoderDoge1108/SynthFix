"""
SynthFix: Baseline Training (SFT / RFT)

SFT = standard cross-entropy on (buggy → fixed) pairs
RFT = reward-weighted policy gradient using symbolic reward

Usage:
    python -m src.train_baseline --model codegen-220m --method sft --dataset data/fixjs ...
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import types as _types
import sys as _sys

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

import numpy as np
import torch
import torch.nn.functional as F
from peft import get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.backends.cudnn.benchmark = True

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from data.dataset import create_dataloaders
from models.reward import compute_reward
from train_synthfix import (
    MODEL_PATHS,
    _compute_codebleu,
    _detect_language,
    _log,
    evaluate,
    get_lora_config,
)


def train_sft(model, tokenizer, train_loader, val_loader, output_dir,
              args, device):
    _log('SFT training')
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for epoch in range(args.epochs):
        model.train()
        eloss = 0.0
        for batch in tqdm(train_loader, desc=f'SFT E{epoch+1}'):
            ids = batch['input_ids'].to(device, non_blocking=True)
            mask = batch['attention_mask'].to(device, non_blocking=True)
            lab = batch['labels'].to(device, non_blocking=True)
            out = model(input_ids=ids, attention_mask=mask, labels=lab)
            opt.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            eloss += out.loss.item()
        sched.step()
        _log(f'  Epoch {epoch+1}/{args.epochs}: '
             f'loss={eloss/len(train_loader):.4f}')

    final = Path(output_dir) / 'final_model'
    final.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    return model


def _per_sample_ce(model, full_ids, full_mask, labels):
    """Compute per-sample CE over unmasked positions.

    Returns (per_sample_ce, valid_mask) where valid_mask marks samples
    with at least one unmasked label token.
    """
    outputs = model(input_ids=full_ids, attention_mask=full_mask)
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    bs, sl_m1, vocab = shift_logits.shape
    per_token = F.cross_entropy(
        shift_logits.view(-1, vocab),
        shift_labels.view(-1),
        reduction='none', ignore_index=-100,
    ).view(bs, sl_m1)
    tok_mask = (shift_labels != -100).float()
    n_tok = tok_mask.sum(1)
    valid = (n_tok > 0).float()
    per_sample = (per_token * tok_mask).sum(1) / n_tok.clamp(min=1)
    return per_sample, valid


def train_rft(model, tokenizer, train_loader, val_loader, output_dir,
              args, device):
    """Reward-weighted fine-tuning (REINFORCE with baseline + SFT anchor).

    Fixes over the previous broken implementation:
      * Correct policy-gradient sign: loss = advantage × CE_on_sampled
        (CE = -log pi, so minimizing advantage×CE increases log pi of
        high-reward actions).
      * Correct label masking: mask only the PROMPT region, not the
        entire ground-truth CLM concat.
      * Moving-average baseline for variance reduction.
      * SFT anchor (ground-truth CE, weighted by 1-beta) keeps the model
        close to the teacher and prevents policy collapse — this is the
        standard "RL+SFT" mixture used in practice.
      * 1-epoch SFT warmup so the policy has a sane starting distribution
        before reward sampling kicks in.
    """
    _log('RFT training (REINFORCE + baseline + SFT anchor)')
    model = model.to(device)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr * 0.5, weight_decay=0.01,
    )

    warmup_epochs = 1 if args.epochs > 1 else 0
    baseline = 0.0
    baseline_momentum = 0.9
    BETA_RL = 0.5  # RL weight; (1-BETA_RL) goes to SFT anchor
    SAMPLE_TEMP = 0.8
    MAX_GEN = 128

    best_val_loss = float('inf')
    best_model_dir = None

    for epoch in range(args.epochs):
        phase = 'SFT' if epoch < warmup_epochs else 'RFT'
        model.train()
        ep_reward = 0.0
        ep_loss = 0.0
        nb = 0

        for batch in tqdm(train_loader, desc=f'{phase} E{epoch+1}/{args.epochs}'):
            sft_ids = batch['input_ids'].to(device, non_blocking=True)
            sft_mask = batch['attention_mask'].to(device, non_blocking=True)
            sft_lbl = batch['labels'].to(device, non_blocking=True)

            if phase == 'SFT':
                out = model(input_ids=sft_ids, attention_mask=sft_mask,
                            labels=sft_lbl)
                loss = out.loss
            else:
                prompt_ids = batch['prompt_input_ids'].to(device,
                                                           non_blocking=True)
                prompt_mask = batch['prompt_attention_mask'].to(device,
                                                                 non_blocking=True)
                fixed_texts = batch['fixed_text']
                prompt_len = prompt_ids.size(1)

                with torch.no_grad():
                    gen_out = model.generate(
                        input_ids=prompt_ids, attention_mask=prompt_mask,
                        max_new_tokens=MAX_GEN, do_sample=True,
                        temperature=SAMPLE_TEMP, top_p=0.95,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                full_ids = gen_out
                full_mask = (full_ids != tokenizer.pad_token_id).long()
                # Mask the prompt portion; keep the generated portion.
                rl_labels = full_ids.clone()
                rl_labels[:, :prompt_len] = -100
                rl_labels[full_mask == 0] = -100

                rewards = []
                for j in range(full_ids.size(0)):
                    gen_text = tokenizer.decode(
                        full_ids[j, prompt_len:], skip_special_tokens=True)
                    r = compute_reward(
                        gen_text, fixed_texts[j],
                        args.lambda_ast, args.lambda_cfg, args.lambda_sem,
                    )
                    rewards.append(float(r))

                rewards_t = torch.tensor(rewards, device=device,
                                          dtype=torch.float32)
                mean_r = rewards_t.mean().item()
                baseline = (baseline_momentum * baseline
                            + (1.0 - baseline_momentum) * mean_r)
                advantages = rewards_t - baseline
                # Normalize to unit std for scale-invariant RL signal
                adv_std = advantages.std().clamp(min=1e-4)
                advantages = advantages / adv_std

                per_sample_ce_rl, valid = _per_sample_ce(
                    model, full_ids, full_mask, rl_labels)
                # loss_rl = E[advantage * CE(sampled)]
                rl_loss = (advantages * per_sample_ce_rl * valid).sum() \
                          / valid.sum().clamp(min=1)

                sft_out = model(input_ids=sft_ids, attention_mask=sft_mask,
                                labels=sft_lbl)
                loss = BETA_RL * rl_loss + (1.0 - BETA_RL) * sft_out.loss
                ep_reward += mean_r

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            ep_loss += loss.item()
            nb += 1

        _log(f'  Epoch {epoch+1}/{args.epochs} [{phase}]: '
             f'loss={ep_loss/max(nb,1):.4f}  '
             f'reward={ep_reward/max(nb,1):.4f}  '
             f'baseline={baseline:.4f}')

        # Validation: SFT-style val loss (same as SFT baseline criterion)
        model.eval()
        v_loss = 0.0
        v_n = 0
        with torch.no_grad():
            for vb in val_loader:
                v_ids = vb['input_ids'].to(device, non_blocking=True)
                v_m = vb['attention_mask'].to(device, non_blocking=True)
                v_l = vb['labels'].to(device, non_blocking=True)
                vout = model(input_ids=v_ids, attention_mask=v_m, labels=v_l)
                v_loss += vout.loss.item()
                v_n += 1
        v_loss /= max(v_n, 1)
        _log(f'  val_loss={v_loss:.4f}')

        ckpt = Path(output_dir) / f'checkpoint_epoch{epoch+1}'
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_model_dir = ckpt
            _log(f'  -> New best (val_loss={v_loss:.4f})')

    if best_model_dir is not None:
        _log(f'Reloading best model from {best_model_dir}')
        from peft import PeftModel
        base = model.base_model.model
        model = PeftModel.from_pretrained(base, str(best_model_dir))

    final = Path(output_dir) / 'final_model'
    final.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='codegen-220m',
                        choices=list(MODEL_PATHS.keys()))
    parser.add_argument('--method', type=str, required=True,
                        choices=['sft', 'rft'])
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--lambda_ast', type=float, default=0.333)
    parser.add_argument('--lambda_cfg', type=float, default=0.333)
    parser.add_argument('--lambda_sem', type=float, default=0.334)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--lora_rank', type=int, default=None)
    parser.add_argument('--grad_checkpoint', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available()
                          else 'cpu')
    model_path = MODEL_PATHS[args.model]
    _log(f'Loading {model_path} → {device}')

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map={'': str(device)})
    model = get_peft_model(model, get_lora_config(args.model, args.lora_rank))

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    _log(f'LoRA r={model.peft_config["default"].r}  '
         f'Params: {trainable:,} / {total_p:,} ({100*trainable/total_p:.2f}%)')

    train_loader, val_loader, test_loader = create_dataloaders(
        args.dataset, tokenizer, args.batch_size, args.max_length,
        num_workers=args.num_workers)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    if args.method == 'sft':
        model = train_sft(model, tokenizer, train_loader, val_loader,
                          args.output, args, device)
    else:
        model = train_rft(model, tokenizer, train_loader, val_loader,
                          args.output, args, device)

    results = evaluate(model, tokenizer, test_loader, device, args.dataset)
    results['model'] = args.model
    results['method'] = args.method
    out_path = Path(args.output) / 'results.json'
    out_path.write_text(json.dumps(results, indent=2))
    _log(f'Results: EM={results["exact_match"]*100:.1f}%  '
         f'CodeBLEU={results["codebleu"]*100:.2f}%')


if __name__ == '__main__':
    main()
