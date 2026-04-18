#!/usr/bin/env python3
"""Aggregate the v12 final-run JSONs into a single report.

Reads:
  artifact/results/twostage/{sft,rft}_{fixjs,sven}.json   (baselines)
  artifact/results/final/sf_{fixjs,sven}_s{42,1337,7}.json (main)
  artifact/results/final/abl_*.json                         (ablations)

Emits:
  artifact/results/final/final_report.json
  artifact/results/final/final_report.md
"""

import json
import os
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('SYNTHFIX_ROOT', str(HERE)))
TWOSTAGE = ROOT / 'results' / 'twostage'
FINAL = ROOT / 'results' / 'final'

DATASETS = ['fixjs', 'sven']
SEEDS = [42, 1337, 7]


def _load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def _fmt_pct(x, nd=2):
    return 'n/a' if x is None else f'{x*100:.{nd}f}%'


def _seed_stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None, None
    mean = statistics.mean(values)
    if len(values) >= 2:
        std = statistics.stdev(values)
    else:
        std = 0.0
    return mean, std, values


def main():
    report = {'baselines': {}, 'synthfix_new': {}, 'ablations': {}}
    md = ['# SynthFix v12 — final results', '']

    # Baselines (reused from last two-stage run)
    for ds in DATASETS:
        sft = _load(TWOSTAGE / f'sft_{ds}.json')
        rft = _load(TWOSTAGE / f'rft_{ds}.json')
        old_sf = _load(TWOSTAGE / f'synthfix_{ds}.json')
        report['baselines'][ds] = {
            'SFT':            {'codebleu': sft['codebleu'] if sft else None,
                               'em': sft['exact_match'] if sft else None},
            'RFT':            {'codebleu': rft['codebleu'] if rft else None,
                               'em': rft['exact_match'] if rft else None},
            'SynthFix(old)':  {'codebleu': old_sf['codebleu'] if old_sf else None,
                               'em': old_sf['exact_match'] if old_sf else None},
        }

    # Main matrix: SynthFix new config, 3 seeds per dataset
    for ds in DATASETS:
        per_seed = {}
        cbs = []
        ems = []
        for s in SEEDS:
            r = _load(FINAL / f'sf_{ds}_s{s}.json')
            if r is None:
                per_seed[s] = None
                continue
            per_seed[s] = {
                'codebleu': r.get('codebleu'),
                'em': r.get('exact_match'),
                'codebleu_greedy': r.get('codebleu_greedy'),
                'reranker_delta_pp': r.get('reranker_delta_pp'),
                'train_time_s': r.get('train_time_s'),
            }
            if r.get('codebleu') is not None:
                cbs.append(r['codebleu'])
            if r.get('exact_match') is not None:
                ems.append(r['exact_match'])
        cb_m, cb_s, _ = _seed_stats(cbs)
        em_m, em_s, _ = _seed_stats(ems)
        report['synthfix_new'][ds] = {
            'per_seed': per_seed,
            'mean_codebleu': cb_m,
            'std_codebleu': cb_s,
            'mean_em': em_m,
            'std_em': em_s,
            'n_seeds': len(cbs),
        }

    # Ablations
    for tag in ('abl_old_lr', 'abl_k4', 'abl_no_rerank'):
        r = _load(FINAL / f'{tag}.json')
        if r is None:
            report['ablations'][tag] = None
        else:
            report['ablations'][tag] = {
                'codebleu': r.get('codebleu'),
                'em': r.get('exact_match'),
                'codebleu_greedy': r.get('codebleu_greedy'),
                'reranker_delta_pp': r.get('reranker_delta_pp'),
                'config': r.get('config'),
            }

    (FINAL / 'final_report.json').write_text(json.dumps(report, indent=2))

    # ── Markdown summary ───────────────────────────────────────────
    md.append('## Main results (test CodeBLEU, deepseek-1.3b)')
    md.append('')
    md.append('| Dataset | SFT | RFT | SynthFix (old) | **SynthFix (v12)** | Δ vs SFT | Δ vs old SynthFix |')
    md.append('|--------|-----|-----|---------------|--------------------|---------|-------------------|')
    for ds in DATASETS:
        b = report['baselines'][ds]
        new = report['synthfix_new'][ds]
        sft_cb = b['SFT']['codebleu']
        rft_cb = b['RFT']['codebleu']
        old_cb = b['SynthFix(old)']['codebleu']
        new_m = new['mean_codebleu']
        new_s = new['std_codebleu']
        new_str = (f'**{new_m*100:.2f}% ± {new_s*100:.2f}%** (n={new["n_seeds"]})'
                   if new_m is not None else 'n/a')
        d_sft = (new_m - sft_cb) * 100 if (new_m is not None and sft_cb is not None) else None
        d_old = (new_m - old_cb) * 100 if (new_m is not None and old_cb is not None) else None
        md.append(f'| {ds} | {_fmt_pct(sft_cb)} | {_fmt_pct(rft_cb)} | '
                  f'{_fmt_pct(old_cb)} | {new_str} | '
                  f'{"n/a" if d_sft is None else f"{d_sft:+.2f}pp"} | '
                  f'{"n/a" if d_old is None else f"{d_old:+.2f}pp"} |')
    md.append('')

    md.append('## Exact Match')
    md.append('| Dataset | SFT | RFT | SynthFix (old) | **SynthFix (v12)** |')
    md.append('|--------|-----|-----|---------------|--------------------|')
    for ds in DATASETS:
        b = report['baselines'][ds]
        new = report['synthfix_new'][ds]
        new_m = new['mean_em']; new_s = new['std_em']
        new_str = (f'**{new_m*100:.2f}% ± {new_s*100:.2f}%**'
                   if new_m is not None else 'n/a')
        md.append(f'| {ds} | {_fmt_pct(b["SFT"]["em"])} | '
                  f'{_fmt_pct(b["RFT"]["em"])} | '
                  f'{_fmt_pct(b["SynthFix(old)"]["em"])} | {new_str} |')
    md.append('')

    def _fmt_pp(x):
        return 'n/a' if x is None else f'{x:+.2f}pp'

    def _fmt_s(x):
        return '—' if x is None else str(int(x))

    md.append('## Per-seed breakdown (SynthFix v12)')
    md.append('| Dataset | Seed | CodeBLEU | EM | Greedy CB | Rerank Δ | Time (s) |')
    md.append('|--------|------|---------|-----|----------|----------|--------|')
    for ds in DATASETS:
        for s in SEEDS:
            ps = report['synthfix_new'][ds]['per_seed'].get(s)
            if ps is None:
                md.append(f'| {ds} | {s} | missing | — | — | — | — |')
                continue
            md.append(f'| {ds} | {s} | {_fmt_pct(ps["codebleu"])} | '
                      f'{_fmt_pct(ps["em"])} | '
                      f'{_fmt_pct(ps["codebleu_greedy"])} | '
                      f'{_fmt_pp(ps.get("reranker_delta_pp"))} | '
                      f'{_fmt_s(ps.get("train_time_s"))} |')
    md.append('')

    md.append('## Ablations (fixjs, seed 42)')
    md.append('| Ablation | CodeBLEU | EM | Greedy CB | Rerank Δ |')
    md.append('|---------|---------|-----|----------|----------|')
    main_ref = report['synthfix_new']['fixjs']['per_seed'].get(42) or {}
    md.append(f'| (main) v12 | {_fmt_pct(main_ref.get("codebleu"))} | '
              f'{_fmt_pct(main_ref.get("em"))} | '
              f'{_fmt_pct(main_ref.get("codebleu_greedy"))} | '
              f'{_fmt_pp(main_ref.get("reranker_delta_pp"))} |')
    for tag, desc in [('abl_old_lr',     'lr=2e-4 (old)'),
                       ('abl_k4',         'rloo_k=4'),
                       ('abl_no_rerank',  'no_rerank=True')]:
        d = report['ablations'].get(tag)
        if d is None:
            md.append(f'| {desc} | missing | — | — | — |')
            continue
        md.append(f'| {desc} | {_fmt_pct(d["codebleu"])} | '
                  f'{_fmt_pct(d["em"])} | '
                  f'{_fmt_pct(d["codebleu_greedy"])} | '
                  f'{_fmt_pp(d.get("reranker_delta_pp"))} |')

    (FINAL / 'final_report.md').write_text('\n'.join(md))
    print('Wrote', FINAL / 'final_report.json')
    print('Wrote', FINAL / 'final_report.md')


if __name__ == '__main__':
    main()
