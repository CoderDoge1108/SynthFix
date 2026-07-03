#!/usr/bin/env python
"""RQ2 (router) + RQ5 (reward-component) ablations on functional metrics.

Each job trains a SynthFix variant fresh (same recipe as the main matrix, only
the ablation switch differs), then runs the matching functional eval. Jobs are
scheduled two-at-a-time across the two GPUs (each job stays on one GPU for both
its train and eval phase). Device is pinned via CUDA_VISIBLE_DEVICES.

Compared against the already-computed full SynthFix results:
  codeflaws deepseek-1.3b: SFT 48 -> SynthFix 59
  pyrepair  deepseek-1.3b: SFT 79 -> SynthFix 100
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / 'artifact'
RUN = ART / 'run_all_experiments.py'
CKPT = ART / 'checkpoints' / 'matrix'
OUT = ART / 'results' / 'artifact_prep' / 'ablation'
OUT.mkdir(parents=True, exist_ok=True)
PY = sys.executable

MODEL = 'deepseek-1.3b'
BS = '16'

DATA = {
    'codeflaws': ('artifact/work/codeflaws_exec', '--use_rich_exec_reward',
                  'eval_functional.py', 'data:artifact/work/codeflaws_exec'),
    'pyrepair': ('data/benchmarks_processed/pyrepair', '--use_rich_reward',
                 'eval_functional_py.py', 'data:data/benchmarks_processed/pyrepair'),
}

# tag -> (dataset, [extra train flags])
JOBS = {
    'rq2_norouter_codeflaws': ('codeflaws', ['--no_router']),
    'rq2_norouter_pyrepair':  ('pyrepair',  ['--no_router']),
    'rq5_noexec_codeflaws':   ('codeflaws', ['--ablate_reward', 'no_exec']),
    'rq5_nostruct_codeflaws': ('codeflaws', ['--ablate_reward', 'no_struct']),
    'rq5_norepair_codeflaws': ('codeflaws', ['--ablate_reward', 'no_repair']),
}


def train_cmd(tag, dataset, extra):
    data, reward_flag, _, _ = DATA[dataset]
    ck = CKPT / f'abl_{tag}'
    cmd = [PY, '-u', str(RUN), '--worker', '--method', 'synthfix',
           '--model_name', MODEL, '--dataset_name', dataset,
           '--data_dir', data, '--gpu', '0', '--batch_size', BS, '--lr', '2e-4',
           '--lora_rank', '16', '--max_new_tokens', '256',
           '--epochs', '4', '--sft_warmup_epochs', '2',
           '--rl_beta', '0.12', '--kl_beta', '0.12', '--rloo_k', '2',
           '--rl_temp', '0.95', '--rl_top_p', '0.95', '--rl_no_repeat_ngram', '3',
           '--select_metric', 'val_codebleu', '--num_rerank_cands', '16',
           reward_flag,
           '--save_ckpt_to', str(ck),
           '--out', str(OUT / f'train_{tag}.json')] + extra
    return cmd, ck


def eval_cmd(tag, dataset, ck):
    data, _, script, _ = DATA[dataset]
    cmd = [PY, '-u', str(ART / script),
           '--sft_ckpt', str(ck / 'checkpoint_epoch2'),
           '--synthfix_ckpt', str(ck / 'final_model'),
           '--data', data, '--split', 'test',
           '--model_name', MODEL, '--gpu', '0', '--K', '16',
           '--out', str(OUT / f'eval_{tag}.json')]
    return cmd


def run_job(tag, gpu):
    dataset, extra = JOBS[tag]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    tcmd, ck = train_cmd(tag, dataset, extra)
    tlog = open(OUT / f'train_{tag}.log', 'w')
    print(f'[abl] TRAIN {tag} on GPU{gpu}', flush=True)
    rc = subprocess.call(tcmd, cwd=str(ROOT), stdout=tlog,
                         stderr=subprocess.STDOUT, env=env)
    tlog.close()
    if rc != 0 or not (ck / 'final_model').is_dir():
        print(f'[abl] TRAIN FAIL {tag} rc={rc}', flush=True)
        return
    ecmd = eval_cmd(tag, dataset, ck)
    elog = open(OUT / f'eval_{tag}.log', 'w')
    print(f'[abl] EVAL  {tag} on GPU{gpu}', flush=True)
    rc = subprocess.call(ecmd, cwd=str(ROOT), stdout=elog,
                         stderr=subprocess.STDOUT, env=env)
    elog.close()
    print(f'[abl] DONE  {tag} rc={rc}', flush=True)


def main():
    import threading
    pending = list(JOBS)
    if len(sys.argv) > 1:
        pending = [t for t in pending if t in sys.argv[1:]]
    # assign jobs round-robin to 2 GPUs via a simple worker-per-GPU queue
    lock = threading.Lock()

    def worker(gpu):
        while True:
            with lock:
                if not pending:
                    return
                tag = pending.pop(0)
            run_job(tag, gpu)

    threads = [threading.Thread(target=worker, args=(g,)) for g in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print('[abl] all ablation jobs complete', flush=True)


if __name__ == '__main__':
    main()
