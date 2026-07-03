#!/usr/bin/env python
"""Train + evaluate the RFT-only baseline across the full model x benchmark
matrix, so Table 1 / Figure 2 can report SFT / RFT / SynthFix on the
deployable metrics (functional pass@1, security-cleared rate).

RFT-only uses the budget-matched baseline config (method=rft, epochs=2,
lr 2e-4, LoRA r=16, rft_rl_temp 0.8). Each job:
  1. trains RFT fresh, saving the adapter to checkpoints/matrix_rft/,
  2. evaluates RFT *greedy* via the existing eval scripts (the --sft_ckpt slot
     does a greedy decode of whatever adapter it is given), so the reported
     `sft_greedy*` field == RFT-only greedy.

GPU-aware scheduler (mirrors run_eval_sweep.py). qwen3-4b uses .venv_qwen3.
Resumable: a job is skipped once its eval JSON exists.
"""
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / 'artifact'
RUN = ART / 'run_all_experiments.py'
CKPT = ART / 'checkpoints' / 'matrix_rft'
OUT = ART / 'results' / 'artifact_prep'
TRAIN_OUT = OUT / 'matrix_rft'
LOGDIR = OUT / 'rft_matrix'
QWEN3_PY = str(ROOT / '.venv_qwen3' / 'bin' / 'python')
for d in (CKPT, TRAIN_OUT, LOGDIR):
    d.mkdir(parents=True, exist_ok=True)

DATASETS = {
    'pyrepair': {
        'data': 'data/benchmarks_processed/pyrepair',
        'eval': 'eval_functional_py.py',
        'evout': 'rft_functional_pyrepair_{m}.json',
        'split': True,
    },
    'codeflaws': {
        'data': 'artifact/work/codeflaws_exec',
        'eval': 'eval_functional.py',
        'evout': 'rft_functional_codeflaws_{m}.json',
        'split': True,
    },
    'sven': {
        'data': 'data/benchmarks_processed/sven',
        'eval': 'eval_security_sven.py',
        'evout': 'rft_security_sven_{m}.json',
        'split': False,
    },
}

MODELS = {
    'deepseek-1.3b': {'bs': 16, 'gc': False},
    'llama3.2-3b':   {'bs': 8,  'gc': False},
    'qwen3-4b':      {'bs': 8,  'gc': True, 'venv': True},
    'codellama-7b':  {'bs': 4,  'gc': True},
    'starcoder2-7b': {'bs': 4,  'gc': True},
}
# heaviest last so the cheap rows finish first
MODEL_ORDER = ['deepseek-1.3b', 'llama3.2-3b', 'qwen3-4b',
               'codellama-7b', 'starcoder2-7b']
DS_ORDER = ['pyrepair', 'sven', 'codeflaws']


def py_for(m):
    return QWEN3_PY if MODELS[m].get('venv') else sys.executable


def ckpt_dir(m, ds):
    return CKPT / f'rft_{m}_{ds}'


def evout_path(m, ds):
    return OUT / DATASETS[ds]['evout'].format(m=m)


def build_job_cmd(m, ds):
    mc = MODELS[m]
    d = DATASETS[ds]
    py = py_for(m)
    ck = ckpt_dir(m, ds)
    train_out = TRAIN_OUT / f'rft_{m}_{ds}.json'

    # Budget-matched to SFT baseline / SynthFix: 2-epoch SFT warmup + 2 RL
    # epochs (epochs=4, sft_warmup_epochs=2). RFT differs from SynthFix only
    # in the router, exec-grounded reward, and best-of-K selection.
    train = [py, '-u', str(RUN), '--worker', '--method', 'rft',
             '--model_name', m, '--dataset_name', ds, '--data_dir', d['data'],
             '--out', str(train_out), '--gpu', '0',
             '--batch_size', str(mc['bs']), '--lr', '2e-4',
             '--lora_rank', '16', '--max_new_tokens', '256',
             '--epochs', '4', '--sft_warmup_epochs', '2', '--rft_rl_temp', '0.8',
             '--save_ckpt_to', str(ck)]
    if mc['gc']:
        train.append('--grad_checkpoint')

    ev = [py, '-u', str(ART / d['eval']),
          '--sft_ckpt', str(ck), '--synthfix_ckpt', str(ck),
          '--data', d['data'], '--model_name', m, '--gpu', '0',
          '--K', '16', '--out', str(evout_path(m, ds))]
    if d['split']:
        ev += ['--split', 'test']

    # train then (only if a checkpoint was produced) eval
    return f'{shlex.join(train)} && {shlex.join(ev)}'


def free_gpus(threshold_mb=2000):
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.used',
             '--format=csv,noheader,nounits'], text=True)
    except Exception:
        return []
    free = []
    for line in out.strip().splitlines():
        idx, mem = [x.strip() for x in line.split(',')]
        if int(mem) < threshold_mb:
            free.append(int(idx))
    return free


def main():
    jobs = [(m, ds) for m in MODEL_ORDER for ds in DS_ORDER
            if not evout_path(m, ds).exists()]
    print(f'[rft] {len(jobs)} jobs queued: '
          + ', '.join(f'{m}/{ds}' for m, ds in jobs), flush=True)

    running = {}      # gpu -> (proc, m, ds, lf, t0)
    pending = list(jobs)

    while pending or running:
        for gpu in list(running):
            proc, m, ds, lf, t0 = running[gpu]
            if proc.poll() is not None:
                lf.close()
                ok = proc.returncode == 0 and evout_path(m, ds).exists()
                print(f'[rft] {"DONE" if ok else "FAIL"} {m}/{ds} '
                      f'rc={proc.returncode} {int(time.time()-t0)}s', flush=True)
                del running[gpu]

        if pending:
            busy = set(running)
            for gpu in free_gpus():
                if gpu in busy or not pending:
                    continue
                m, ds = pending.pop(0)
                lf = open(LOGDIR / f'{m}_{ds}.log', 'w')
                cmd = build_job_cmd(m, ds)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                p = subprocess.Popen(['bash', '-c', cmd], cwd=str(ROOT),
                                     stdout=lf, stderr=subprocess.STDOUT,
                                     env=env)
                running[gpu] = (p, m, ds, lf, time.time())
                print(f'[rft] launch {m}/{ds} on GPU{gpu} pid={p.pid}',
                      flush=True)
                busy.add(gpu)
                time.sleep(10)

        time.sleep(20)

    print('[rft] all jobs complete', flush=True)


if __name__ == '__main__':
    main()
