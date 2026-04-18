#!/usr/bin/env python3
"""Two-stage orchestrator v2 - uses nvidia-smi to gate dispatch.

Stage 2 only launches on a GPU once actual memory usage is low AND the
corresponding SFT ckpt is on disk.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('SYNTHFIX_ROOT', str(HERE)))
RESULTS = Path(os.environ.get(
    'SYNTHFIX_RESULTS_TWOSTAGE', str(HERE / 'results' / 'twostage')))
RESULTS.mkdir(parents=True, exist_ok=True)

_DATA_ROOT = Path(os.environ.get(
    'SYNTHFIX_DATA_DIR', str(HERE / 'data' / 'processed')))
DATA = {
    'fixjs': str(_DATA_ROOT / 'fixjs'),
    'sven':  str(_DATA_ROOT / 'sven'),
}
_SFT_ROOT = Path(os.environ.get(
    'SYNTHFIX_SFT_DIR', str(HERE / 'results' / 'sft_foundation')))
SFT_CKPT = {
    'fixjs': str(_SFT_ROOT / 'fixjs'),
    'sven':  str(_SFT_ROOT / 'sven'),
}
STAGE2 = [
    ('rft_fixjs',      'rft',       'fixjs', 2),
    ('synthfix_fixjs', 'synthfix',  'fixjs', 2),
    ('rft_sven',       'rft',       'sven',  2),
    ('synthfix_sven',  'synthfix',  'sven',  2),
]

# Skip tags whose .json already exists (already succeeded or in progress).
def already_done_or_running(tag):
    out = RESULTS / f'{tag}.json'
    return out.is_file()

def ckpt_ready(ds):
    return (Path(SFT_CKPT[ds]) / 'adapter_config.json').is_file()

def gpu_mem_used_mb(gpu):
    """Return used MB on the physical GPU."""
    try:
        r = subprocess.check_output(
            ['nvidia-smi', f'--id={gpu}',
             '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            text=True).strip()
        return int(r)
    except Exception:
        return 999_999

# Track any currently-running synthfix/rft we dispatched
def find_our_jobs():
    out = subprocess.check_output(['pgrep', '-af', 'run_all_experiments.py'],
                                   text=True)
    return out

def build_cmd(tag, method, ds, epochs, gpu):
    out = RESULTS / f'{tag}.json'
    cmd = [
        'python', str(HERE / 'run_all_experiments.py'),
        '--worker', '--method', method, '--gpu', '0',
        '--data_dir', DATA[ds],
        '--out', str(out),
        '--model_name', 'deepseek-1.3b',
        '--dataset_name', ds,
        '--batch_size', '16',
        '--epochs', str(epochs),
        '--lora_rank', '16',
        '--lr', '2e-4',
        '--max_new_tokens', '256',
        '--init_from_ckpt', SFT_CKPT[ds],
        '--seed', '42',
    ]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    return cmd, env

MEM_FREE_THRESHOLD_MB = 8000  # only dispatch if GPU < 8GB used

def main():
    pending = [j for j in STAGE2 if not already_done_or_running(j[0])]
    running = {}  # gpu -> (tag, proc, logf)

    def P(msg):
        print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)

    P(f'Pending: {[j[0] for j in pending]}')

    while pending or running:
        # Reap finished
        for gpu in list(running.keys()):
            tag, proc, logf = running[gpu]
            if proc.poll() is not None:
                logf.close()
                P(f'GPU {gpu} done: {tag} (rc={proc.returncode})')
                del running[gpu]

        # Dispatch new jobs to empty GPUs that are ALSO low on memory
        for gpu in (0, 1):
            if gpu in running:
                continue
            if not pending:
                continue
            used = gpu_mem_used_mb(gpu)
            if used > MEM_FREE_THRESHOLD_MB:
                continue
            # Find a pending job whose SFT ckpt is ready
            idx = None
            for i, (tag, method, ds, ep) in enumerate(pending):
                if ckpt_ready(ds):
                    idx = i
                    break
            if idx is None:
                continue
            tag, method, ds, ep = pending.pop(idx)
            logp = RESULTS / f'{tag}.log'
            cmd, env = build_cmd(tag, method, ds, ep, gpu)
            logf = open(logp, 'w')
            P(f'GPU {gpu} (used={used}MB) launch: {tag}')
            proc = subprocess.Popen(cmd, env=env, stdout=logf,
                                     stderr=subprocess.STDOUT)
            running[gpu] = (tag, proc, logf)
        time.sleep(15)

    P('All stage-2 jobs done.')

if __name__ == '__main__':
    main()
