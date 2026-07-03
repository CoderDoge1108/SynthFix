"""Resumable job queue for the SynthFix artifact result matrix.

Runs the {SFT, RFT, SynthFix} x {pyrepair, CodeFlaws, SVEN} x {models}
matrix using the reported artifact recipe:
  * SFT  : 2 epochs pure SFT.
  * RFT  : 2 epochs (1 SFT warmup + REINFORCE), rft_temp=0.8.
  * SynthFix: 4 epochs = 2 SFT warmup + 2 router-gated RLOO, KL anchor,
    rich reward (js/python) or rich+exec reward (codeflaws C),
    checkpoint selection by val_codebleu, K=16 reranker.

It schedules two jobs at a time (one per GPU), is resumable (skips any job
whose output JSON already exists), and logs each job separately.

Usage:
  python artifact/queue_artifact.py            # run everything missing
  python artifact/queue_artifact.py --dry_run  # print the plan only
  python artifact/queue_artifact.py --models qwen2.5-coder-1.5b deepseek-1.3b
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # project root
ART = ROOT / 'artifact'
RUN = ART / 'run_all_experiments.py'
OUT_DIR = ART / 'results' / 'artifact_prep' / 'matrix'
CKPT_DIR = ART / 'checkpoints' / 'matrix'
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ── Datasets ────────────────────────────────────────────────────────────
DATASETS = {
    'fixjs':     {'data': 'data/benchmarks_sampled/fixjs', 'reward': 'rich',
                  'mnt': 256},
    'codeflaws': {'data': 'artifact/work/codeflaws_exec', 'reward': 'exec',
                  'mnt': 256},
    'sven':      {'data': 'data/benchmarks_processed/sven', 'reward': 'rich',
                  'mnt': 256},
    # Python execution-repair benchmark (replaces FixJS): MBPP-injected
    # train/val + MBPP held-out & QuixBugs test. Functional pass@1 metric.
    'pyrepair':  {'data': 'data/benchmarks_processed/pyrepair', 'reward': 'rich',
                  'mnt': 256},
}

# ── Models. Base env = transformers 4.45 (sys.executable). qwen3-4b needs
#    transformers>=4.51 + torch>=2.5, so it runs in the isolated .venv_qwen3.
QWEN3_PY = str(ROOT / '.venv_qwen3' / 'bin' / 'python')
MODELS = {
    'deepseek-1.3b':        {'bs': 16, 'gc': False},
    'llama3.2-3b':          {'bs': 8,  'gc': False},
    'codellama-7b':         {'bs': 4,  'gc': True},
    'starcoder2-7b':        {'bs': 4,  'gc': True},
    'qwen3-4b':             {'bs': 8,  'gc': True, 'py': QWEN3_PY},
}

# Priority order for resumable scheduling; Qwen3 uses an isolated environment.
MODEL_ORDER = ['deepseek-1.3b', 'llama3.2-3b',
               'starcoder2-7b', 'codellama-7b', 'qwen3-4b']
# Active artifact benchmarks use execution- or security-grounded metrics.
# FixJS results are retained only as supplementary overlap-metric diagnostics.
DATASET_ORDER = ['pyrepair', 'sven', 'codeflaws']
METHOD_ORDER = ['sft', 'rft', 'synthfix']

# Optional precomputed results can be added here as
# {(model, dataset, method): "filename.json"}. The public artifact does not
# require precomputed matrix outputs; missing entries are recomputed.
PRECOMPUTED = {}


def out_path(model, dataset, method):
    return OUT_DIR / f'{method}_{model}_{dataset}.json'


def build_cmd(model, dataset, method, gpu):
    ds = DATASETS[dataset]
    mc = MODELS[model]
    out = out_path(model, dataset, method)
    log = OUT_DIR / f'{method}_{model}_{dataset}.log'
    py = mc.get('py', sys.executable)
    cmd = [py, '-u', str(RUN), '--worker',
           '--method', method, '--model_name', model,
           '--dataset_name', dataset, '--data_dir', ds['data'],
           '--out', str(out), '--gpu', '0',
           '--batch_size', str(mc['bs']), '--lr', '2e-4',
           '--lora_rank', '16', '--max_new_tokens', str(ds['mnt'])]
    if mc['gc']:
        cmd.append('--grad_checkpoint')
    if method in ('sft', 'rft'):
        cmd += ['--epochs', '2']
        if method == 'rft':
            cmd += ['--rft_rl_temp', '0.8']
    else:  # synthfix artifact recipe
        cmd += ['--epochs', '4', '--sft_warmup_epochs', '2',
                '--rl_beta', '0.12', '--kl_beta', '0.12', '--rloo_k', '2',
                '--rl_temp', '0.95', '--rl_top_p', '0.95',
                '--rl_no_repeat_ngram', '3',
                '--select_metric', 'val_codebleu',
                '--num_rerank_cands', '16',
                '--save_ckpt_to', str(CKPT_DIR / f'synthfix_{model}_{dataset}')]
        if ds['reward'] == 'exec':
            cmd.append('--use_rich_exec_reward')
        else:
            cmd.append('--use_rich_reward')
    return cmd, out, log


def make_jobs(only_models):
    jobs = []
    for model in MODEL_ORDER:
        if only_models and model not in only_models:
            continue
        for dataset in DATASET_ORDER:
            for method in METHOD_ORDER:
                out = out_path(model, dataset, method)
                # Reuse a precomputed result if present.
                leg = PRECOMPUTED.get((model, dataset, method))
                if leg:
                    legp = ART / 'results' / 'artifact_prep' / leg
                    if legp.exists() and not out.exists():
                        out.write_text(legp.read_text())
                if out.exists():
                    continue
                jobs.append((model, dataset, method))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--models', nargs='*', default=None)
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    jobs = make_jobs(args.models)
    print(f'[queue] {len(jobs)} jobs to run:')
    for j in jobs:
        print('   ', '/'.join(j))
    if args.dry_run or not jobs:
        return

    free = list(args.gpus)
    running = {}  # gpu -> (proc, job, log, t0)
    pending = list(jobs)
    env_base = dict(os.environ)

    def launch(job, gpu):
        cmd, out, log = build_cmd(*job, gpu)
        env = dict(env_base)
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        lf = open(log, 'w')
        p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT, env=env)
        print(f'[queue] START {"/".join(job)} on GPU{gpu} (pid {p.pid}) '
              f'-> {log.name}', flush=True)
        return (p, job, lf, time.time())

    while pending or running:
        while free and pending:
            gpu = free.pop(0)
            running[gpu] = launch(pending.pop(0), gpu)
        time.sleep(20)
        for gpu, (p, job, lf, t0) in list(running.items()):
            if p.poll() is not None:
                lf.close()
                ok = (p.returncode == 0) and out_path(*job).exists()
                dt = (time.time() - t0) / 60
                print(f'[queue] {"DONE" if ok else "FAIL"} {"/".join(job)} '
                      f'on GPU{gpu} rc={p.returncode} ({dt:.0f}min)', flush=True)
                del running[gpu]
                free.append(gpu)
    print('[queue] all jobs finished.', flush=True)


if __name__ == '__main__':
    main()
