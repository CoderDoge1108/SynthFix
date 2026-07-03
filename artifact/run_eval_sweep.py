#!/usr/bin/env python
"""GPU-aware confirmation-eval sweep for the SynthFix artifact.

Runs the headline (functional / security) evals for every (model, benchmark)
whose SynthFix checkpoint is ready, comparing the 2-epoch SFT warmup
(checkpoint_epoch2) against the final SynthFix model (final_model) via best-of-K.

- pyrepair  -> eval_functional_py.py   (Python execution pass@1)
- codeflaws -> eval_functional.py      (C execution pass@1)
- sven      -> eval_security_sven.py   (semgrep vulnerability-cleared rate)

Auto-detects free GPUs from nvidia-smi so it never collides with a training
job still occupying a device. qwen3-4b inference is routed through .venv_qwen3.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / 'artifact'
CKPT = ART / 'checkpoints' / 'matrix'
OUT = ART / 'results' / 'artifact_prep'
QWEN3_PY = str(ROOT / '.venv_qwen3' / 'bin' / 'python')

# model order: deepseek already fully confirmed, so it is omitted here.
MODELS = ['llama3.2-3b', 'starcoder2-7b', 'codellama-7b', 'qwen3-4b']

BENCH = {
    'pyrepair': {
        'script': 'eval_functional_py.py',
        'data': 'data/benchmarks_processed/pyrepair',
        'extra': ['--split', 'test'],
        'out': 'functional_pyrepair_{m}.json',
    },
    'codeflaws': {
        'script': 'eval_functional.py',
        'data': 'artifact/work/codeflaws_exec',
        'extra': ['--split', 'test'],
        'out': 'functional_codeflaws_{m}.json',
    },
    'sven': {
        'script': 'eval_security_sven.py',
        'data': 'data/benchmarks_processed/sven',
        'extra': [],
        'out': 'security_sven_{m}.json',
    },
}


def py_for(model):
    return QWEN3_PY if model == 'qwen3-4b' else sys.executable


def ckpt_ready(model, ds):
    ck = CKPT / f'synthfix_{model}_{ds}'
    # require the matrix result JSON too: it is written only after the SynthFix
    # training job fully completes, so we never grab a checkpoint mid-training.
    done = (OUT / 'matrix' / f'synthfix_{model}_{ds}.json').exists()
    return (done and (ck / 'final_model').is_dir()
            and (ck / 'checkpoint_epoch2').is_dir())


def out_path(model, ds):
    return OUT / BENCH[ds]['out'].format(m=model)


def build_cmd(model, ds, gpu):
    ck = CKPT / f'synthfix_{model}_{ds}'
    b = BENCH[ds]
    # IMPORTANT: pin the device via CUDA_VISIBLE_DEVICES and always use cuda:0
    # inside the eval. Loading a model with device_map={'':'cuda:1'} and
    # generating on a non-zero CUDA index produces EMPTY generations for some
    # LLaMA-arch models (observed on codellama/starcoder2), silently yielding
    # 0 solved. Remapping the physical GPU to cuda:0 avoids this entirely.
    cmd = [py_for(model), '-u', str(ART / b['script']),
           '--sft_ckpt', str(ck / 'checkpoint_epoch2'),
           '--synthfix_ckpt', str(ck / 'final_model'),
           '--data', b['data'],
           '--model_name', model,
           '--gpu', '0',
           '--K', '16',
           '--out', str(out_path(model, ds))] + b['extra']
    return cmd


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
    jobs = []
    for m in MODELS:
        for ds in BENCH:
            if out_path(m, ds).exists():
                continue
            jobs.append((m, ds))
    print(f'[sweep] {len(jobs)} eval jobs queued: '
          + ', '.join(f'{m}/{ds}' for m, ds in jobs), flush=True)

    running = {}  # gpu -> (proc, model, ds, logfile, t0)
    pending = list(jobs)

    while pending or running:
        # reap finished
        for gpu in list(running):
            proc, m, ds, lf, t0 = running[gpu]
            if proc.poll() is not None:
                lf.close()
                ok = proc.returncode == 0 and out_path(m, ds).exists()
                dt = int(time.time() - t0)
                print(f'[sweep] {"DONE" if ok else "FAIL"} {m}/{ds} '
                      f'rc={proc.returncode} {dt}s', flush=True)
                del running[gpu]

        # launch on free gpus
        if pending:
            busy = set(running)
            for gpu in free_gpus():
                if gpu in busy:
                    continue
                # find a ready pending job
                pick = None
                for j in pending:
                    if ckpt_ready(*j):
                        pick = j
                        break
                if pick is None:
                    break
                pending.remove(pick)
                m, ds = pick
                lfp = OUT / (BENCH[ds]['out'].format(m=m).replace('.json', '.log'))
                lf = open(lfp, 'w')
                cmd = build_cmd(m, ds, gpu)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=lf,
                                     stderr=subprocess.STDOUT, env=env)
                running[gpu] = (p, m, ds, lf, time.time())
                print(f'[sweep] launch {m}/{ds} on GPU{gpu} pid={p.pid}',
                      flush=True)
                busy.add(gpu)
                time.sleep(8)  # stagger model loads

        time.sleep(20)

    print('[sweep] all eval jobs complete', flush=True)


if __name__ == '__main__':
    main()
