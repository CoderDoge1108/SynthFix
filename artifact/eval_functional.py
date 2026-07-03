"""Functional-correctness evaluation for CodeFlaws.

Headline metric = test-suite pass rate, not CodeBLEU. We compile each
candidate patch and run it against the per-bug Codeflaws test suite.

Leak-free selection protocol:
  * SELECT a candidate using *public* tests only (input-pos*).
  * REPORT that candidate's pass rate on *heldout* tests
    (heldout-input-pos*), which were never used for selection.

Methods compared:
  * SFT greedy           : single greedy patch (one shot).
  * SynthFix greedy      : single greedy patch from the SynthFix adapter.
  * SynthFix best-of-K   : K diverse candidates, select by public-test
                           pass rate (repair-effect static score breaks
                           ties), report heldout pass of the selection.
  * SynthFix oracle@K    : upper bound — best heldout pass among the K
                           (diagnostic only, uses heldout to select).

Reported per method:
  * solved@1  : fraction of bugs where the chosen patch passes ALL its
                heldout tests (a functionally correct repair).
  * mean heldout pass rate over bugs.

Usage:
  CUDA_VISIBLE_DEVICES=1 python artifact/eval_functional.py \
    --sft_ckpt artifact/checkpoints/matrix/sft_deepseek-1.3b_codeflaws/final_model \
    --synthfix_ckpt artifact/checkpoints/matrix/synthfix_deepseek-1.3b_codeflaws/final_model \
    --data artifact/work/codeflaws_exec \
    --out artifact/results/artifact_prep/functional_codeflaws.json \
    --gpu 0 --K 16 --max_bugs 391
"""
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch

SRC = Path(__file__).resolve().parent / 'src'
sys.path.insert(0, str(SRC))

from data.dataset import create_dataloaders  # noqa: E402
from models.inference import (generate_k_candidates,  # noqa: E402
                              set_feature_language)
from models.exec_reward import (_compile, _run_test,  # noqa: E402
                                CodeflawsTestCase, discover_main_name)
from models.repair_effect import repair_effect_score  # noqa: E402
from train_synthfix import MODEL_PATHS, _detect_language  # noqa: E402


def _list_tests(bug_dir: Path, in_prefix: str, out_prefix: str,
                cap: int) -> List[CodeflawsTestCase]:
    out = []
    if not bug_dir.exists():
        return out
    for inp in sorted(bug_dir.glob(f'{in_prefix}*'))[:cap]:
        idx = inp.name[len(in_prefix):]
        of = bug_dir / f'{out_prefix}{idx}'
        if of.exists():
            out.append(CodeflawsTestCase(input_file=inp, output_file=of,
                                         name=f'{in_prefix}{idx}'))
    return out


@dataclass
class CandEval:
    text: str
    compiled: bool
    public_pass: float
    heldout_pass: float
    repair: float


def _eval_candidate(text: str, bug_dir: Path, main: str,
                    public: List[CodeflawsTestCase],
                    heldout: List[CodeflawsTestCase],
                    buggy: str, compile_cache: dict) -> CandEval:
    import tempfile
    key = hash(text)
    if key in compile_cache:
        return compile_cache[key]
    rep = float(repair_effect_score(text, buggy, 'c'))
    with tempfile.TemporaryDirectory(prefix='sfx_fn_') as tmp:
        work = Path(tmp)
        ok, _ = _compile(text, work, main)
        if not ok:
            res = CandEval(text, False, 0.0, 0.0, rep)
            compile_cache[key] = res
            return res
        exe = work / main

        def _rate(tests):
            if not tests:
                return 0.0
            p = sum(1 for t in tests if _run_test(exe, t)[0])
            return p / len(tests)

        res = CandEval(text, True, _rate(public), _rate(heldout), rep)
    compile_cache[key] = res
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sft_ckpt', required=True)
    ap.add_argument('--synthfix_ckpt', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model_name', default='deepseek-1.3b')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--K', type=int, default=16)
    ap.add_argument('--max_new_tokens', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--max_bugs', type=int, default=391)
    ap.add_argument('--public_cap', type=int, default=6)
    ap.add_argument('--heldout_cap', type=int, default=12)
    ap.add_argument('--split', choices=['val', 'test'], default='test',
                    help='Which split to evaluate. Use val for leak-free '
                         'checkpoint selection; test for the final report.')
    ap.add_argument('--greedy_only', action='store_true',
                    help='Only evaluate the --sft_ckpt adapter under greedy '
                         'decoding (records sft_greedy). Skips the K-candidate '
                         'best-of-K pass: ~K x faster for a single greedy '
                         'pass@1 number (used for the RFT-only baseline).')
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    torch.manual_seed(42)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf = MODEL_PATHS[args.model_name]
    print(f'[fn] base {hf} on GPU {args.gpu}', flush=True)
    tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id

    base = AutoModelForCausalLM.from_pretrained(
        hf, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map={'': str(device)})
    model = PeftModel.from_pretrained(base, args.sft_ckpt, adapter_name='sft')
    if not args.greedy_only:
        model.load_adapter(args.synthfix_ckpt, adapter_name='synthfix')
    model.eval()

    lang = _detect_language(args.data)
    set_feature_language(lang)
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data, tok, args.batch_size, 512, num_workers=2, shuffle_seed=42)
    eval_loader = val_loader if args.split == 'val' else test_loader
    print(f'[fn] split={args.split}', flush=True)

    import random as _random
    rng = _random.Random(42)

    agg = {
        'sft_greedy': {'solved': 0, 'sum': 0.0},
        'synthfix_greedy': {'solved': 0, 'sum': 0.0},
        'synthfix_random': {'solved': 0, 'sum': 0.0},
        'synthfix_bestofk': {'solved': 0, 'sum': 0.0},
        'synthfix_oracle': {'solved': 0, 'sum': 0.0},
    }
    n = 0
    t0 = time.time()

    for batch in eval_loader:
        pids = batch['prompt_input_ids'].to(device, non_blocking=True)
        pmask = batch['prompt_attention_mask'].to(device, non_blocking=True)
        cur = pids.size(0)

        model.set_adapter('sft')
        prompt_len = pids.size(1)
        with torch.no_grad():
            sft_out = model.generate(
                input_ids=pids, attention_mask=pmask,
                max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=pad_id)
        sft_texts = [tok.decode(sft_out[j, prompt_len:],
                                skip_special_tokens=True).strip()[:4000]
                     for j in range(cur)]

        if not args.greedy_only:
            model.set_adapter('synthfix')
            conts, _lp, _tp, gfl, ifl = generate_k_candidates(
                model, tok, pids, pmask, K=args.K,
                max_new_tokens=args.max_new_tokens, pad_id=pad_id,
                output_scores=False)

        for j in range(cur):
            if n >= args.max_bugs:
                break
            td = batch.get('test_dirs', [''] * cur)[j]
            if not td or not os.path.isdir(td):
                continue
            bug_dir = Path(td)
            main = discover_main_name(bug_dir) or 'prog'
            public = _list_tests(bug_dir, 'input-pos', 'output-pos',
                                 args.public_cap)
            heldout = _list_tests(bug_dir, 'heldout-input-pos',
                                  'heldout-output-pos', args.heldout_cap)
            if not public or not heldout:
                continue
            buggy = batch['buggy_text'][j]
            cache = {}

            sft_e = _eval_candidate(sft_texts[j], bug_dir, main, public,
                                    heldout, buggy, cache)

            # Fast path: greedy-only (RFT-only baseline). Record sft_greedy
            # and skip the expensive K-candidate best-of-K pass.
            if args.greedy_only:
                agg['sft_greedy']['sum'] += sft_e.heldout_pass
                agg['sft_greedy']['solved'] += (
                    1 if sft_e.heldout_pass >= 0.999 else 0)
                n += 1
                if n % 25 == 0:
                    print(f"[fn] {n} bugs  sft_greedy="
                          f"{agg['sft_greedy']['solved']}/{n}"
                          f"({agg['sft_greedy']['sum']/n*100:.1f}%)  "
                          f"{time.time()-t0:.0f}s", flush=True)
                continue
            # SynthFix candidates (skip identity copies).
            sf_cands = [c for c, isi in zip(conts[j], ifl[j]) if not isi]
            sf_greedy_text = None
            for c, g in zip(conts[j], gfl[j]):
                if g:
                    sf_greedy_text = c
                    break
            cand_evals = [_eval_candidate(c, bug_dir, main, public, heldout,
                                          buggy, cache) for c in sf_cands]
            if not cand_evals:
                continue

            sf_greedy_e = next(
                (e for e in cand_evals if e.text == sf_greedy_text),
                cand_evals[0])
            # Best-of-K selection with a GREEDY FLOOR (leak-free: only
            # public tests + static repair score are consulted).
            #   * Keep the greedy patch unless another candidate STRICTLY
            #     beats it on public-test pass rate. This guarantees
            #     best-of-K never regresses below greedy when public tests
            #     track heldout correctness.
            #   * Among candidates that strictly beat greedy on public
            #     tests, take the highest public pass; break ties by the
            #     static repair-effect score, then compiled flag.
            max_pub = max(e.public_pass for e in cand_evals)
            if max_pub <= sf_greedy_e.public_pass + 1e-9:
                best = sf_greedy_e
            else:
                tied = [e for e in cand_evals
                        if abs(e.public_pass - max_pub) < 1e-9]
                best = max(tied, key=lambda e: (e.repair,
                                                1.0 if e.compiled else 0.0))
            oracle = max(cand_evals, key=lambda e: e.heldout_pass)
            # Inference-ablation baseline: pick a random candidate from the
            # SAME K pool (no symbolic guidance). Isolates the value of
            # symbolic-feature-guided selection vs the candidate pool itself.
            rand_e = rng.choice(cand_evals)

            for kk, e in (('sft_greedy', sft_e),
                          ('synthfix_greedy', sf_greedy_e),
                          ('synthfix_random', rand_e),
                          ('synthfix_bestofk', best),
                          ('synthfix_oracle', oracle)):
                agg[kk]['sum'] += e.heldout_pass
                agg[kk]['solved'] += 1 if e.heldout_pass >= 0.999 else 0
            n += 1
            if n % 25 == 0:
                cur_line = ' '.join(
                    f"{k}={v['solved']}/{n}({v['sum']/n*100:.1f}%)"
                    for k, v in agg.items())
                print(f'[fn] {n} bugs  {cur_line}  '
                      f'{time.time()-t0:.0f}s', flush=True)
        if n >= args.max_bugs:
            break

    out = {'method': 'SynthFix-Functional', 'model': args.model_name,
           'dataset': 'codeflaws', 'n_bugs': n, 'K': args.K,
           'metrics': {}}
    for k, v in agg.items():
        out['metrics'][k] = {
            'solved_at_1': v['solved'] / max(n, 1),
            'mean_heldout_pass': v['sum'] / max(n, 1),
            'solved_count': v['solved'],
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print('\n[fn] ─── FUNCTIONAL SUMMARY (heldout) ───', flush=True)
    for k, v in out['metrics'].items():
        print(f'[fn] {k:20s} solved@1={v["solved_at_1"]*100:.1f}%  '
              f'mean_pass={v["mean_heldout_pass"]*100:.1f}%', flush=True)
    print(f'[fn] n_bugs={n}  saved -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
