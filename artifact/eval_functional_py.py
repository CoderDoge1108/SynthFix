"""Functional-correctness eval for the Python execution-repair benchmark.

Mirrors eval_functional.py (Codeflaws C) but for inline Python assert
suites (MBPP). Leak-free best-of-K: candidate selection consults only the
PUBLIC test subset; the reported solved@1 uses the held-out subset.

  solved@1 = chosen patch passes ALL held-out tests.
  best-of-K keeps SFT greedy unless a candidate strictly beats it on
  public pass-rate (greedy floor -> never regresses).
"""
import argparse
import ast
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parent / 'src'
sys.path.insert(0, str(SRC))

from data.dataset import create_dataloaders  # noqa: E402
from models.inference import generate_k_candidates, set_feature_language  # noqa: E402
from train_synthfix import MODEL_PATHS, _compute_codebleu  # noqa: E402

HARNESS = r'''
import sys
_PASS = 0
{setup}
{code}
__TESTS = {tests!r}
for _t in __TESTS:
    try:
        exec(_t, globals()); _PASS += 1
    except Exception:
        pass
print("RESULT", _PASS, len(__TESTS))
'''


def run_pass(code, tests, setup='', timeout=10):
    if not tests:
        return 0, 0
    prog = HARNESS.format(setup=setup or '', code=code, tests=tests)
    try:
        r = subprocess.run([sys.executable, '-c', prog],
                           capture_output=True, text=True, timeout=timeout)
        for line in r.stdout.splitlines()[::-1]:
            if line.startswith('RESULT'):
                _, p, t = line.split()
                return int(p), int(t)
    except Exception:
        pass
    return 0, len(tests)


def parses(code):
    try:
        ast.parse(code); return True
    except Exception:
        return False


def split_tests(tests):
    """public (selection) / heldout (reporting)."""
    if len(tests) <= 1:
        return tests, tests          # tiny: unavoidable overlap
    k = (len(tests) + 1) // 2
    return tests[:k], tests[k:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sft_ckpt', required=True)
    ap.add_argument('--synthfix_ckpt', required=True)
    ap.add_argument('--data', default='data/benchmarks_processed/pyrepair')
    ap.add_argument('--split', default='test')
    ap.add_argument('--out', required=True)
    ap.add_argument('--model_name', default='deepseek-1.3b')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--K', type=int, default=16)
    ap.add_argument('--max_new_tokens', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=4)
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf = MODEL_PATHS[args.model_name]
    tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id
    base = AutoModelForCausalLM.from_pretrained(
        hf, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map={'': str(device)})
    model = PeftModel.from_pretrained(base, args.sft_ckpt, adapter_name='sft')
    model.load_adapter(args.synthfix_ckpt, adapter_name='synthfix')
    model.eval()
    set_feature_language('python')

    meta = json.loads(Path(args.data, f'{args.split}.json').read_text())
    _, _, test_loader = create_dataloaders(
        args.data, tok, args.batch_size, 512, num_workers=2, shuffle_seed=42)

    def greedy(pids, pmask):
        plen = pids.size(1)
        with torch.no_grad():
            out = model.generate(input_ids=pids, attention_mask=pmask,
                                  max_new_tokens=args.max_new_tokens,
                                  do_sample=False, pad_token_id=pad_id)
        return [tok.decode(out[j, plen:], skip_special_tokens=True).strip()[:4000]
                for j in range(out.size(0))]

    sft_solved = sf_solved = oracle_solved = 0
    sft_cb = sf_cb = 0.0
    n = 0; gidx = 0; t0 = time.time()
    for batch in test_loader:
        pids = batch['prompt_input_ids'].to(device, non_blocking=True)
        pmask = batch['prompt_attention_mask'].to(device, non_blocking=True)
        cur_bs = pids.size(0)
        model.set_adapter('sft')
        sft_texts = greedy(pids, pmask)
        model.set_adapter('synthfix')
        conts, logps, _, gflags, iflags = generate_k_candidates(
            model, tok, pids, pmask, K=args.K,
            max_new_tokens=args.max_new_tokens, pad_id=pad_id,
            output_scores=False)
        for j in range(cur_bs):
            m = meta[gidx]; gidx += 1; n += 1
            tests = m['tests']; setup = m.get('test_setup', '')
            ref = m['fixed']
            pub, held = split_tests(tests)
            cands = [sft_texts[j]] + conts[j]
            ident = [False] + iflags[j]
            logp = [0.0] + logps[j]

            def heldout_solved(code):
                p, t = run_pass(code, held, setup)
                return t > 0 and p == t

            # SFT greedy
            sft_ok = heldout_solved(cands[0])
            sft_solved += int(sft_ok)
            sft_cb += _compute_codebleu([cands[0]], [ref], lang='python')

            # best-of-K with greedy floor on PUBLIC tests
            sft_pub = run_pass(cands[0], pub, setup)[0]
            best_i, best_key = 0, (sft_pub, 0, 0.0)
            for ci in range(1, len(cands)):
                if ident[ci] or not parses(cands[ci]):
                    continue
                pp = run_pass(cands[ci], pub, setup)[0]
                key = (pp, 1, logp[ci])
                if pp > sft_pub and key > best_key:
                    best_key, best_i = key, ci
            chosen = cands[best_i]
            sf_ok = heldout_solved(chosen)
            sf_solved += int(sf_ok)
            sf_cb += _compute_codebleu([chosen], [ref], lang='python')

            # oracle: any candidate that solves heldout
            oracle_solved += int(any(
                heldout_solved(c) for ci, c in enumerate(cands)
                if not ident[ci]))
        print(f'[pyfunc] {n} (sft={sft_solved} sf={sf_solved} '
              f'oracle={oracle_solved}) {time.time()-t0:.0f}s', flush=True)

    res = {
        'model': args.model_name, 'data': args.data, 'split': args.split,
        'K': args.K, 'n': n,
        'sft_greedy_solved': sft_solved,
        'synthfix_bestK_solved': sf_solved,
        'oracle_solved': oracle_solved,
        'sft_solved_rate': sft_solved / max(1, n),
        'synthfix_solved_rate': sf_solved / max(1, n),
        'oracle_solved_rate': oracle_solved / max(1, n),
        'codebleu': {'sft_greedy': sft_cb / max(1, n) * 100,
                     'synthfix_bestK': sf_cb / max(1, n) * 100},
    }
    if sft_solved:
        res['synthfix_vs_sft_rel_pct'] = \
            (sf_solved - sft_solved) / sft_solved * 100
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, 'w'), indent=2)
    print('\n=== PYTHON FUNCTIONAL ===')
    print(json.dumps({k: v for k, v in res.items()
                      if k != 'codebleu'}, indent=2))


if __name__ == '__main__':
    main()
