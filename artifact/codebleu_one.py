#!/usr/bin/env python
"""Standalone greedy-CodeBLEU eval for a single LoRA checkpoint.

Replicates the baseline (non-SynthFix) greedy eval path in
run_all_experiments.run_worker so the number matches what the in-process
RFT eval would have produced -- used to fill the two CodeFlaws cells whose
budget-matched RFT runs were interrupted during the in-process CodeBLEU pass.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'artifact'))
sys.path.insert(0, str(ROOT / 'artifact' / 'src'))

from data.dataset import create_dataloaders  # noqa: E402
from train_synthfix import (MODEL_PATHS, _compute_codebleu,  # noqa: E402
                            _detect_language)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--max_new_tokens', type=int, default=256)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    torch.manual_seed(args.seed)
    hf = MODEL_PATHS[args.model_name]
    lang = _detect_language(args.data_dir)
    P = lambda *a: print(*a, flush=True)
    P(f'[cb] {args.model_name} lang={lang} ckpt={args.ckpt}')

    tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        hf, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map={'': str(device)})
    model = PeftModel.from_pretrained(base, args.ckpt)
    model.eval()

    _, _, test_loader = create_dataloaders(
        args.data_dir, tok, args.bs, 512, num_workers=2,
        shuffle_seed=args.seed)
    pad_id = tok.pad_token_id

    all_gen, all_ref = [], []
    with torch.no_grad():
        for bi, batch in enumerate(test_loader):
            pids = batch['prompt_input_ids'].to(device)
            pmask = batch['prompt_attention_mask'].to(device)
            plen = pids.size(1)
            gen = model.generate(
                input_ids=pids, attention_mask=pmask,
                max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=pad_id)
            decoded = [tok.decode(g[plen:], skip_special_tokens=True
                                  ).strip()[:2000] for g in gen]
            for j, gt in enumerate(decoded):
                all_gen.append(gt)
                all_ref.append(batch['fixed_text'][j].strip()[:2000])
            if (bi + 1) % 10 == 0:
                P(f'[cb] {len(all_gen)} samples')

    cb = _compute_codebleu(all_gen, all_ref, lang=lang)
    em = sum(1 for g, r in zip(all_gen, all_ref) if g == r)
    out = {'method': 'RFT', 'model': args.model_name,
           'dataset': Path(args.data_dir).name, 'language': lang,
           'codebleu': cb, 'exact_match': em / len(all_gen),
           'em_count': em, 'total': len(all_gen),
           'ckpt': args.ckpt, 'note': 'standalone greedy codebleu'}
    Path(args.out).write_text(json.dumps(out, indent=2))
    P(f'[cb] DONE {args.model_name}: CodeBLEU={cb*100:.2f}%  '
      f'EM={em}/{len(all_gen)}  -> {args.out}')


if __name__ == '__main__':
    main()
