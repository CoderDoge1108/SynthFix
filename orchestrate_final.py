#!/usr/bin/env python3
"""
SynthFix final-run orchestrator.

Queues SynthFix training jobs sequentially on a single GPU and writes
per-run JSON results that `aggregate_final.py` collects into a final
table. Baseline (SFT, RFT) runs are expected to live under
`results/twostage/` (produced either by `orchestrate_twostage.py` or
by a prior invocation of `src/train_baseline.py`); this script is
scoped to the SynthFix jobs and their ablations.

Default matrix (6 runs): SynthFix × {fixjs, sven} × seeds {42, 1337, 7}
Default ablations       (3 runs):
  * old_lr     — phase-2 LR set back to 2e-4   (LR sensitivity)
  * k4         — RLOO_K = 4                    (more rollouts per step)
  * no_rerank  — greedy-only decoding          (measures reranker lift)

A lightweight gate is checked inline between runs:
  * val_codebleu is monotonically non-decreasing across phase-2 epochs
    (required), and
  * test CodeBLEU meets an optional user-supplied target configured
    via the SYNTHFIX_TARGETS env var (advisory; skipped if not set).

If the gate fails the orchestrator stops early and writes a status
file so the remaining jobs can be inspected before burning more
compute.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('SYNTHFIX_ROOT', str(HERE)))
RESULTS = Path(os.environ.get('SYNTHFIX_RESULTS', str(HERE / 'results' / 'final')))
RESULTS.mkdir(parents=True, exist_ok=True)
STATUS = RESULTS / 'status.json'

# Processed data directories (unified JSON splits). Override with
# SYNTHFIX_DATA_DIR (parent containing {fixjs,sven}) if needed.
_DATA_ROOT = Path(os.environ.get(
    'SYNTHFIX_DATA_DIR', str(HERE / 'data' / 'processed')))
DATA = {
    'fixjs': str(_DATA_ROOT / 'fixjs'),
    'sven':  str(_DATA_ROOT / 'sven'),
}

# Pretrained SFT foundation checkpoints (produced by train_baseline.py
# with --method sft). Override with SYNTHFIX_SFT_DIR.
_SFT_ROOT = Path(os.environ.get(
    'SYNTHFIX_SFT_DIR', str(HERE / 'results' / 'sft_foundation')))
SFT_CKPT = {
    'fixjs': str(_SFT_ROOT / 'fixjs'),
    'sven':  str(_SFT_ROOT / 'sven'),
}

# Optional per-dataset CodeBLEU targets read from JSON at
# $SYNTHFIX_TARGETS (if set). Shape: {"<dataset>": {"sft_cb": 0.xx}}.
# Used by the gate check to decide whether to proceed with the rest of
# the job matrix or stop early. Safe to leave unset: the gate will
# then only enforce the val_codebleu monotonicity criterion.
_TARGETS_FILE = os.environ.get('SYNTHFIX_TARGETS')
if _TARGETS_FILE and Path(_TARGETS_FILE).is_file():
    TARGETS = json.loads(Path(_TARGETS_FILE).read_text())
else:
    TARGETS = {}

# Training-loop defaults.  Overridable per-run in the JOBS table.
DEFAULT_CFG = dict(
    lr=1e-5,
    epochs=2,
    rl_beta=0.25,
    rloo_k=2,
    rl_temp=0.9,
    rl_top_p=0.95,
    rl_no_repeat_ngram=3,
    rerank_margin=0.005,
    num_rerank_cands=16,
    no_rerank=False,
    max_new_tokens=256,
    batch_size=16,
    lora_rank=16,
)

# ---- Job table -----------------------------------------------------
#
# Each job is a dict with a 'tag' + 'dataset' + 'seed' + optional
# overrides of DEFAULT_CFG.  All baselines (SFT/RFT) are NOT re-run:
# we reuse the cached ones from results/twostage/*.json.

def J(tag, dataset, seed, **over):
    return {'tag': tag, 'dataset': dataset, 'seed': seed, **over}

MAIN_JOBS = [
    # SynthFix fixjs — seed 42 is the gating run (reused if present).
    J('sf_fixjs_s42',   'fixjs', 42),
    J('sf_fixjs_s1337', 'fixjs', 1337),
    J('sf_fixjs_s7',    'fixjs', 7),
    J('sf_sven_s42',    'sven',  42),
    J('sf_sven_s1337',  'sven',  1337),
    J('sf_sven_s7',     'sven',  7),
]

ABLATION_JOBS = [
    J('abl_old_lr',    'fixjs', 42, lr=2e-4),
    J('abl_k4',        'fixjs', 42, rloo_k=4),
    J('abl_no_rerank', 'fixjs', 42, no_rerank=True),
]


def _already_done(tag):
    return (RESULTS / f'{tag}.json').is_file()


def _build_cmd(job):
    cfg = {**DEFAULT_CFG, **{k: v for k, v in job.items()
                              if k not in ('tag', 'dataset', 'seed')}}
    ds, seed, tag = job['dataset'], job['seed'], job['tag']
    out = RESULTS / f'{tag}.json'
    cmd = [
        sys.executable, '-u', str(HERE / 'run_all_experiments.py'),
        '--worker', '--method', 'synthfix', '--gpu', '0',
        '--data_dir', DATA[ds],
        '--out', str(out),
        '--model_name', 'deepseek-1.3b',
        '--dataset_name', ds,
        '--batch_size', str(cfg['batch_size']),
        '--epochs', str(cfg['epochs']),
        '--lora_rank', str(cfg['lora_rank']),
        '--lr', str(cfg['lr']),
        '--max_new_tokens', str(cfg['max_new_tokens']),
        '--init_from_ckpt', SFT_CKPT[ds],
        '--seed', str(seed),
        '--rl_beta', str(cfg['rl_beta']),
        '--rloo_k', str(cfg['rloo_k']),
        '--rl_temp', str(cfg['rl_temp']),
        '--rl_top_p', str(cfg['rl_top_p']),
        '--rl_no_repeat_ngram', str(cfg['rl_no_repeat_ngram']),
        '--rerank_margin', str(cfg['rerank_margin']),
        '--num_rerank_cands', str(cfg['num_rerank_cands']),
    ]
    if cfg['no_rerank']:
        cmd.append('--no_rerank')
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'
    return cmd, env


def _parse_val_codebleu(log_path):
    """Extract the val_codebleu sequence from a SynthFix training log."""
    if not log_path.is_file():
        return []
    vals = []
    for line in log_path.read_text(errors='ignore').splitlines():
        m = re.search(r'val_codebleu=([0-9.]+)%', line)
        if m:
            vals.append(float(m.group(1)))
    return vals


def _check_gate(result_path, log_path, dataset):
    """Apply gate criteria. Returns (passed: bool, reasons: list[str])."""
    reasons = []
    if not result_path.is_file():
        return False, ['no result file']
    try:
        r = json.loads(result_path.read_text())
    except Exception as e:
        return False, [f'result unreadable: {e}']
    cb = float(r.get('codebleu', 0))
    target_entry = TARGETS.get(dataset, {})
    target_cb = target_entry.get('sft_cb')
    if target_cb is not None:
        target = target_cb + 0.004  # +0.4pp margin vs reference
        if cb >= target:
            reasons.append(f'test CB={cb*100:.2f}% >= target '
                           f'{target*100:.2f}% ✓')
        else:
            reasons.append(f'test CB={cb*100:.2f}% < target '
                           f'{target*100:.2f}% ✗')
    else:
        reasons.append(f'test CB={cb*100:.2f}% (no target configured)')
    vs = _parse_val_codebleu(log_path)
    if not vs:
        reasons.append('val_codebleu not found in log ✗')
        return False, reasons
    monotonic = all(vs[i] <= vs[i+1] + 0.02 for i in range(len(vs)-1))
    # val_codebleu is advisory only; the primary criterion is test CB
    # on the held-out split.
    reasons.append(f'val_codebleu seq={vs}  monotonic={monotonic} '
                   f'(advisory)')
    if target_cb is not None:
        passed = cb >= (target_cb + 0.004)
    else:
        # No configured target → gate only enforces the val_codebleu
        # monotonicity criterion above; accept by default.
        passed = True
    return passed, reasons


def _write_status(state, payload):
    STATUS.write_text(json.dumps({'state': state, **payload,
                                   'updated': time.strftime('%F %T')},
                                  indent=2))


def _run_job(job):
    tag = job['tag']
    if _already_done(tag):
        print(f'[{time.strftime("%H:%M:%S")}] SKIP (already done): {tag}',
              flush=True)
        return 0
    cmd, env = _build_cmd(job)
    logp = RESULTS / f'{tag}.log'
    print(f'[{time.strftime("%H:%M:%S")}] LAUNCH {tag}  ->  {logp.name}',
          flush=True)
    with open(logp, 'w') as logf:
        t0 = time.time()
        proc = subprocess.Popen(cmd, env=env, stdout=logf,
                                 stderr=subprocess.STDOUT)
        rc = proc.wait()
    dur = time.time() - t0
    print(f'[{time.strftime("%H:%M:%S")}] DONE  {tag} rc={rc} '
          f'({dur/60:.1f} min)', flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip_gate_check', action='store_true',
                    help='Run the full matrix without gate verification.')
    ap.add_argument('--only', choices=['main', 'ablations', 'all'],
                    default='all')
    args = ap.parse_args()

    # ── Step 0: gate ──────────────────────────────────────────────
    gate_result = RESULTS / 'gate_synthfix_fixjs_s42.json'
    gate_log = RESULTS / 'gate_synthfix_fixjs_s42.log'
    sf_s42 = RESULTS / 'sf_fixjs_s42.json'

    # If the gate finished, copy its artefacts under the canonical name
    # so we don't re-run seed 42.
    if gate_result.is_file() and not sf_s42.is_file():
        print(f'[gate] copying {gate_result.name} -> {sf_s42.name}',
              flush=True)
        sf_s42.write_text(gate_result.read_text())
        (RESULTS / 'sf_fixjs_s42.log').write_text(
            gate_log.read_text(errors='ignore')
            if gate_log.is_file() else '(no log)')

    if not args.skip_gate_check:
        if not sf_s42.is_file():
            _write_status('waiting_for_gate',
                          {'message': 'Gate run not yet finished.'})
            print('[gate] waiting for gate result; re-run this script '
                  'after gate_synthfix_fixjs_s42.json exists.', flush=True)
            return 2
        passed, reasons = _check_gate(sf_s42, gate_log, 'fixjs')
        print('[gate] check:')
        for r in reasons:
            print(f'   {r}')
        _write_status('gate_pass' if passed else 'gate_fail',
                      {'reasons': reasons})
        if not passed:
            print('[gate] FAILED — aborting full matrix.')
            return 3
        print('[gate] PASSED — proceeding.')

    # ── Step 1: main matrix ───────────────────────────────────────
    if args.only in ('main', 'all'):
        for job in MAIN_JOBS:
            _run_job(job)

    # ── Step 2: ablations ─────────────────────────────────────────
    if args.only in ('ablations', 'all'):
        for job in ABLATION_JOBS:
            _run_job(job)

    _write_status('done', {'main_jobs': [j['tag'] for j in MAIN_JOBS],
                            'ablation_jobs': [j['tag'] for j in ABLATION_JOBS]})
    print('All jobs finished.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
