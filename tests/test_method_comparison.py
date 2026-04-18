"""
SynthFix: Method Comparison (Multi-GPU)

SFT (GPU 0) || RFT (GPU 1) in parallel, then SynthFix (GPU 0).
Paper hyperparameters: BS=16, epochs=10, StepLR(gamma=0.95).

Usage:
    python tests/test_method_comparison.py                 # orchestrator
    python tests/test_method_comparison.py --method sft ... # worker
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import types as _types
import sys as _sys

if 'torch._dynamo' not in _sys.modules:
    _m = _types.ModuleType('torch._dynamo')
    _m.disable = lambda fn=None, recursive=True: fn if fn else (lambda f: f)
    _m.graph_break = lambda: None
    _m.is_compiling = lambda: False
    _mc = _types.ModuleType('torch._dynamo.config')
    _mc.suppress_errors = True
    _m.config = _mc
    _sys.modules['torch._dynamo'] = _m
    _sys.modules['torch._dynamo.config'] = _mc

import torch
import torch.nn.functional as F

_P = lambda *a, **kw: print(*a, **kw, flush=True)

SRC_DIR = Path(__file__).resolve().parent.parent / 'src'
sys.path.insert(0, str(SRC_DIR))

MODEL_NAME = 'codegen-220m'
EPOCHS = 10
BATCH_SIZE = 16
MAX_LEN = 256
TRAIN_SAMPLES = 2000
TEST_SAMPLES = 200
LR = 5e-5
NUM_WORKERS = 4


def prepare_data():
    from data.process_benchmarks import process_fixjs
    tmpbase = Path(tempfile.mkdtemp())
    raw = Path(os.environ.get(
        'SYNTHFIX_RAW_FIXJS',
        str(Path(__file__).resolve().parent.parent / 'data' / 'raw_benchmarks' / 'fixjs'),
    ))
    assert raw.exists(), (
        f'Need real FixJS data at {raw}. Set SYNTHFIX_RAW_FIXJS env var.')
    process_fixjs(str(raw), tmpbase)
    data_dir = tmpbase / 'fixjs'
    for split, n in [('train', TRAIN_SAMPLES), ('val', 100),
                     ('test', TEST_SAMPLES)]:
        f = data_dir / f'{split}.json'
        data = json.loads(f.read_text())
        f.write_text(json.dumps(data[:n]))
    return str(data_dir)


def run_worker(method: str, gpu: int, data_dir: str, out_path: str):
    """Single-method training worker."""
    import numpy as np
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from data.dataset import create_dataloaders
    from models.reward import compute_reward
    from models.router import (RouterModel, FeatureNormalizer,
                               extract_code_features)
    from train_synthfix import (get_lora_config, MODEL_PATHS,
                                _compute_codebleu, _compute_crystalbleu,
                                LossTracker)

    torch.backends.cudnn.benchmark = True
    device = torch.device(f'cuda:{gpu}')

    def load_model():
        tok = AutoTokenizer.from_pretrained(MODEL_PATHS[MODEL_NAME])
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        mdl = AutoModelForCausalLM.from_pretrained(
            MODEL_PATHS[MODEL_NAME], torch_dtype=torch.bfloat16,
            device_map={'': str(device)})
        mdl = get_peft_model(mdl, get_lora_config(MODEL_NAME))
        return mdl, tok

    def eval_model(model, tokenizer, test_loader):
        model.eval()
        em = total = 0
        all_gen, all_ref = [], []
        with torch.no_grad():
            for batch in test_loader:
                ids = batch['input_ids'].to(device, non_blocking=True)
                mask = batch['attention_mask'].to(device, non_blocking=True)
                gen = model.generate(
                    input_ids=ids, attention_mask=mask,
                    max_new_tokens=128, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id)
                for j, g in enumerate(gen):
                    gt = tokenizer.decode(g, skip_special_tokens=True).strip()[:2000]
                    rt = batch['fixed_text'][j].strip()[:2000]
                    all_gen.append(gt)
                    all_ref.append(rt)
                    if gt == rt:
                        em += 1
                    total += 1
        cb = _compute_codebleu(all_gen, all_ref, lang='javascript')
        crb = _compute_crystalbleu(all_gen, all_ref, lang='javascript')
        return {'exact_match': em / total if total else 0,
                'codebleu': cb, 'crystalbleu': crb,
                'total': total, 'em_count': em}

    model, tokenizer = load_model()
    pad_id = tokenizer.pad_token_id
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir, tokenizer, BATCH_SIZE, MAX_LEN, num_workers=NUM_WORKERS)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    _P(f'[{method.upper()}] GPU {gpu} | BS={BATCH_SIZE} | '
       f'LoRA r={model.peft_config["default"].r}')
    _P(f'[{method.upper()}] Params: {trainable:,}/{total_p:,} | '
       f'{TRAIN_SAMPLES} train, {TEST_SAMPLES} test, {EPOCHS} epochs')
    t0 = time.time()

    if method == 'sft':
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                       weight_decay=0.01)
        sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1,
                                                  gamma=0.95)
        model.train()
        for epoch in range(EPOCHS):
            eloss = 0
            for batch in train_loader:
                ids = batch['input_ids'].to(device, non_blocking=True)
                mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)
                out = model(input_ids=ids, attention_mask=mask, labels=labels)
                optimizer.zero_grad()
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                eloss += out.loss.item()
            sched.step()
            _P(f'  [{method.upper()}] Epoch {epoch+1}/{EPOCHS}: '
               f'loss={eloss/len(train_loader):.4f}')

    elif method == 'rft':
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR * 0.1)
        model.train()
        for epoch in range(EPOCHS):
            ereward = 0; nb = 0
            for batch in train_loader:
                ids = batch['input_ids'].to(device, non_blocking=True)
                mask = batch['attention_mask'].to(device, non_blocking=True)
                bs = ids.size(0)
                with torch.no_grad():
                    gen_ids = model.generate(
                        input_ids=ids, attention_mask=mask,
                        max_new_tokens=128, do_sample=True,
                        temperature=0.8, pad_token_id=pad_id)
                rewards = [compute_reward(
                    tokenizer.decode(gen_ids[j], skip_special_tokens=True),
                    batch['fixed_text'][j]) for j in range(bs)]
                ereward += sum(rewards) / len(rewards)

                mean_r = np.mean(rewards)
                optimizer.zero_grad()
                n_valid = 0
                for j in range(min(len(gen_ids), len(ids))):
                    gi = gen_ids[j:j+1]; si = ids[j:j+1]; sm = mask[j:j+1]
                    if gi.shape[1] <= si.shape[1]:
                        continue
                    rl = gi.clone()
                    rl[:, :si.shape[1]] = -100
                    rm = torch.ones_like(gi, dtype=sm.dtype)
                    rm[:, :sm.shape[1]] = sm
                    if (rl != -100).sum() == 0:
                        continue
                    out = model(input_ids=gi, attention_mask=rm, labels=rl)
                    advantage = rewards[j] - mean_r
                    loss = -out.loss * advantage
                    loss.backward()
                    n_valid += 1
                if n_valid > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                nb += 1
            _P(f'  [{method.upper()}] Epoch {epoch+1}/{EPOCHS}: '
               f'avg_reward={ereward/max(nb,1):.4f}')

    elif method == 'synthfix':
        from train_synthfix import train_synthfix as _train_synthfix

        router = RouterModel(input_size=5, hidden_size=64).to(device)

        class _Args:
            lr = LR
            router_lr = 1e-3
            epochs = EPOCHS
            lambda_ast = 0.333
            lambda_cfg = 0.333
            lambda_sem = 0.334
            entropy_coeff = 0.01

        model, router = _train_synthfix(
            model, tokenizer, router,
            train_loader, val_loader,
            tempfile.mkdtemp(), _Args(), device)

    train_time = time.time() - t0
    _P(f'  [{method.upper()}] Training done in {train_time:.1f}s, evaluating...')

    results = eval_model(model, tokenizer, test_loader)
    results['method'] = method.upper() if method != 'synthfix' else 'SynthFix'
    results['train_time_s'] = train_time
    results['gpu'] = gpu
    results['peak_gpu_mb'] = torch.cuda.max_memory_allocated(device) / 1e6
    results['config'] = {
        'batch_size': BATCH_SIZE, 'epochs': EPOCHS,
        'lora_rank': model.peft_config['default'].r if hasattr(model, 'peft_config') else None,
        'train_samples': TRAIN_SAMPLES, 'num_workers': NUM_WORKERS,
    }

    _P(f'  [{method.upper()}] EM={results["exact_match"]*100:.1f}%  '
       f'CodeBLEU={results["codebleu"]*100:.2f}%  '
       f'CrystalBLEU={results["crystalbleu"]*100:.2f}%  '
       f'time={train_time:.0f}s  '
       f'peak_VRAM={results["peak_gpu_mb"]:.0f}MB')

    Path(out_path).write_text(json.dumps(results, indent=2))
    _P(f'  [{method.upper()}] Saved: {out_path}')


# ── Orchestrator ────────────────────────────────────────────────────────

def main():
    _P('=' * 70)
    _P('  SynthFix: Method Comparison (SFT vs RFT vs SynthFix)')
    _P(f'  Model: {MODEL_NAME}  |  Dataset: FixJS')
    _P(f'  Train: {TRAIN_SAMPLES}  |  Test: {TEST_SAMPLES}  |  '
       f'Epochs: {EPOCHS}  |  BS: {BATCH_SIZE}')
    _P(f'  GPUs: 0 + 1  |  Parallel: SFT(GPU0) || RFT(GPU1), '
       f'then SynthFix(GPU0)')
    _P('=' * 70)

    t0 = time.time()
    data_dir = prepare_data()
    _P(f'  Data prepared: {data_dir}')

    tmpdir = Path(tempfile.mkdtemp())
    rfiles = {'SFT': str(tmpdir / 'sft.json'),
              'RFT': str(tmpdir / 'rft.json'),
              'SynthFix': str(tmpdir / 'synthfix.json')}

    py = sys.executable
    script = str(Path(__file__).resolve())

    def spawn(method, gpu, out):
        cmd = [py, '-u', script, '--method', method, '--gpu', str(gpu),
               '--data_dir', data_dir, '--out', out]
        return subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)

    _P('\n  Phase 1: SFT (GPU 0) || RFT (GPU 1)')
    p_sft = spawn('sft', 0, rfiles['SFT'])
    p_rft = spawn('rft', 1, rfiles['RFT'])
    p_sft.wait()
    _P(f'  SFT done (exit {p_sft.returncode})')
    p_rft.wait()
    _P(f'  RFT done (exit {p_rft.returncode})')

    _P('\n  Phase 2: SynthFix (GPU 0)')
    p_sf = spawn('synthfix', 0, rfiles['SynthFix'])
    p_sf.wait()
    _P(f'  SynthFix done (exit {p_sf.returncode})')

    total_time = time.time() - t0

    results = {}
    for name, fp in rfiles.items():
        p = Path(fp)
        if p.exists():
            results[name] = json.loads(p.read_text())
        else:
            _P(f'  WARNING: {name} results missing')
            results[name] = {'exact_match': 0, 'codebleu': 0,
                             'crystalbleu': 0}

    _P('\n' + '=' * 70)
    _P('  RESULTS')
    _P('=' * 70)
    _P(f'  {"Method":<12} {"EM":>8} {"CodeBLEU":>12} '
       f'{"CrystalBLEU":>14} {"Time":>8} {"VRAM_MB":>10}')
    _P('-' * 70)
    for n in ['SFT', 'RFT', 'SynthFix']:
        r = results[n]
        _P(f'  {n:<12} {r["exact_match"]*100:>7.1f}% '
           f'{r["codebleu"]*100:>11.2f}% '
           f'{r["crystalbleu"]*100:>13.2f}%  '
           f'{r.get("train_time_s",0):>6.0f}s'
           f'{r.get("peak_gpu_mb",0):>10.0f}')

    sft_cb = results['SFT']['codebleu']
    rft_cb = results['RFT']['codebleu']
    sf_cb = results['SynthFix']['codebleu']
    best_bl = max(sft_cb, rft_cb)

    _P(f'\n  SynthFix vs best baseline:')
    _P(f'    CodeBLEU:    {(sf_cb - best_bl)*100:+.2f}pp')
    _P(f'    EM:          '
       f'{(results["SynthFix"]["exact_match"]-max(results["SFT"]["exact_match"],results["RFT"]["exact_match"]))*100:+.1f}pp')

    if sf_cb > best_bl:
        _P('\n  >>> SynthFix OUTPERFORMS baselines on CodeBLEU <<<')
    _P(f'\n  Wall time: {total_time/60:.1f} min')

    out_path = Path(__file__).parent / 'comparison_results.json'
    out_path.write_text(json.dumps(results, indent=2))
    _P(f'  Saved: {out_path}')
    _P('=' * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default=None,
                        choices=['sft', 'rft', 'synthfix'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    if args.method:
        run_worker(args.method, args.gpu, args.data_dir, args.out)
    else:
        main()
