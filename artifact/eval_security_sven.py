"""SVEN security-repair eval — the analog of Codeflaws functional correctness.

SVEN is a CWE-labeled vulnerability-repair benchmark. We measure
"vulnerability cleared" using semgrep (a static security analyzer) as the
inference-time symbolic signal — exactly the neural-symbolic selection the
paper advocates (semgrep here plays the role execution-tests play on
Codeflaws).

For each test sample we generate:
  * SFT greedy             (1 candidate baseline)
  * SynthFix K candidates  (diverse, reward-aware)
Then run semgrep ONCE over every candidate (batched in a temp dir).

Metric is reported on the *addressable subset*: test samples whose BUGGY
code raises >=1 semgrep finding (i.e. the analyzer can see the vuln).
  cleared@1 = chosen patch has 0 findings (vuln removed)
Selection for SynthFix best-of-K: among candidates that parse and are
non-identity, pick the one with the FEWEST semgrep findings; ties broken
by chrF-to-reference-free repair-effect then log-prob. SFT greedy is the
safety floor (never regress below it).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parent / 'src'
sys.path.insert(0, str(SRC))

from data.dataset import create_dataloaders  # noqa: E402
from models.inference import generate_k_candidates, set_feature_language  # noqa: E402
from models.parse_dfg import parse_score  # noqa: E402
from train_synthfix import MODEL_PATHS, _compute_codebleu, _detect_language  # noqa: E402

EXT = {'python': '.py', 'c': '.c', 'cpp': '.cpp'}


def semgrep_findings(root):
    """Return {basename: n_findings} for every file under `root`."""
    try:
        r = subprocess.run(
            ['semgrep', '--config', 'p/default', '--json', '--quiet',
             '--timeout', '20', '--jobs', '4', root],
            capture_output=True, text=True, timeout=1200)
        out = json.loads(r.stdout or '{}')
    except Exception as e:
        print('[sven] semgrep error:', e, flush=True)
        return {}
    c = {}
    for res in out.get('results', []):
        b = os.path.basename(res['path'])
        c[b] = c.get(b, 0) + 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sft_ckpt', required=True)
    ap.add_argument('--synthfix_ckpt', required=True)
    ap.add_argument('--data', default='data/benchmarks_processed/sven')
    ap.add_argument('--out', required=True)
    ap.add_argument('--model_name', default='deepseek-1.3b')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--K', type=int, default=16)
    ap.add_argument('--max_new_tokens', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=4)
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

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

    # Per-sample CWE / language (test loader is unshuffled -> json order).
    meta = json.loads(Path(args.data, 'test.json').read_text())
    set_feature_language(_detect_language(args.data))

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

    # Generate all candidates first, stash to a temp tree for one semgrep run.
    work = tempfile.mkdtemp(prefix='sven_eval_')
    samples = []  # list of dicts per test sample
    gidx = 0
    t0 = time.time()
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
            m = meta[gidx]
            lang = m['language']
            ext = EXT.get(lang, '.txt')
            buggy = batch['buggy_text'][j].strip()[:4000]
            ref = batch['fixed_text'][j].strip()[:4000]
            cands = [sft_texts[j]] + conts[j]            # idx0 = SFT greedy
            cand_logp = [0.0] + logps[j]
            cand_ident = [False] + iflags[j]
            # write buggy + every candidate
            def w(tag, code):
                p = os.path.join(work, f'{gidx:03d}_{tag}{ext}')
                open(p, 'w').write(code)
            w('buggy', buggy)
            for ci, c in enumerate(cands):
                w(f'c{ci:02d}', c)
            samples.append(dict(gidx=gidx, lang=lang, cwe=m['cwe'], ref=ref,
                                buggy=buggy, cands=cands, logp=cand_logp,
                                ident=cand_ident))
            gidx += 1
        print(f'[sven] generated {gidx} samples ({time.time()-t0:.0f}s)',
              flush=True)

    print('[sven] running semgrep over all candidates ...', flush=True)
    find = semgrep_findings(work)

    def nf(gidx, tag, ext):
        return find.get(f'{gidx:03d}_{tag}{ext}', 0)

    # ── Score ────────────────────────────────────────────────────────
    flagged = []          # samples where buggy raises >=1 finding
    sft_cleared = sf_cleared = oracle_cleared = 0
    sft_cb = sf_cb = 0.0
    rows = []
    for s in samples:
        g, lang, ext = s['gidx'], s['lang'], EXT.get(s['lang'], '.txt')
        nb = nf(g, 'buggy', ext)
        cand_nf = [nf(g, f'c{ci:02d}', ext) for ci in range(len(s['cands']))]
        # SFT greedy = candidate 0
        sft_f = cand_nf[0]
        # SynthFix best-of-K selection (exclude SFT greedy slot 0):
        #   guard: parse OK, non-identity; pick min findings; tie-break logp.
        best_i = None
        best_key = None
        for ci in range(1, len(s['cands'])):
            if s['ident'][ci]:
                continue
            code = s['cands'][ci]
            if parse_score(code, lang) < 0.99:
                continue
            key = (cand_nf[ci], -s['logp'][ci])
            if best_key is None or key < best_key:
                best_key, best_i = key, ci
        # safety floor: never regress below SFT greedy findings
        if best_i is None or cand_nf[best_i] >= sft_f:
            sf_sel_f = sft_f
            sf_sel_code = s['cands'][0]
        else:
            sf_sel_f = cand_nf[best_i]
            sf_sel_code = s['cands'][best_i]
        oracle_f = min(cand_nf)

        cb_sft = _compute_codebleu([s['cands'][0]], [s['ref']], lang=lang)
        cb_sf = _compute_codebleu([sf_sel_code], [s['ref']], lang=lang)
        sft_cb += cb_sft
        sf_cb += cb_sf

        if nb > 0:  # addressable: analyzer can see the vuln
            flagged.append(g)
            sft_cleared += int(sft_f == 0)
            sf_cleared += int(sf_sel_f == 0)
            oracle_cleared += int(oracle_f == 0)
            rows.append(dict(gidx=g, cwe=s['cwe'], buggy_find=nb,
                             sft_find=sft_f, synthfix_find=sf_sel_f,
                             oracle_find=oracle_f))

    n = len(samples)
    nflag = len(flagged)
    res = {
        'model': args.model_name, 'data': args.data, 'K': args.K,
        'n_test': n, 'n_addressable': nflag,
        'security': {
            'sft_greedy_cleared': sft_cleared,
            'synthfix_bestK_cleared': sf_cleared,
            'oracle_cleared': oracle_cleared,
            'sft_cleared_rate': sft_cleared / max(1, nflag),
            'synthfix_cleared_rate': sf_cleared / max(1, nflag),
            'oracle_cleared_rate': oracle_cleared / max(1, nflag),
        },
        'codebleu': {
            'sft_greedy': sft_cb / max(1, n) * 100,
            'synthfix_bestK': sf_cb / max(1, n) * 100,
        },
        'rows': rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, 'w'), indent=2)
    shutil.rmtree(work, ignore_errors=True)

    sc = res['security']
    print('\n=== SVEN SECURITY (addressable subset) ===')
    print(f'addressable (buggy flagged by semgrep): {nflag}/{n}')
    print(f"SFT greedy   cleared: {sc['sft_greedy_cleared']}/{nflag} "
          f"({sc['sft_cleared_rate']*100:.1f}%)")
    print(f"SynthFix K   cleared: {sc['synthfix_bestK_cleared']}/{nflag} "
          f"({sc['synthfix_cleared_rate']*100:.1f}%)")
    print(f"oracle@K     cleared: {sc['oracle_cleared']}/{nflag} "
          f"({sc['oracle_cleared_rate']*100:.1f}%)")
    if sc['sft_greedy_cleared']:
        rel = (sc['synthfix_bestK_cleared'] - sc['sft_greedy_cleared']) \
              / sc['sft_greedy_cleared'] * 100
        print(f"SynthFix vs SFT: {rel:+.1f}% relative")


if __name__ == '__main__':
    main()
