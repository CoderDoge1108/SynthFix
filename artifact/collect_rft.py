#!/usr/bin/env python
"""Aggregate RFT-only deployable metrics + CodeBLEU across the matrix and
print them next to the existing SFT / SynthFix numbers, ready to paste into
Table 1, Table 2, and the figure script.
"""
import json
from pathlib import Path

R = Path(__file__).resolve().parent / 'results' / 'artifact_prep'
MODELS = ['deepseek-1.3b', 'llama3.2-3b', 'qwen3-4b',
          'codellama-7b', 'starcoder2-7b']
# existing (sft/synthfix) functional files key deepseek as 'deepseek'
ALT = {'deepseek-1.3b': 'deepseek'}


def _load(name):
    p = R / name
    return json.load(open(p)) if p.exists() else None


def pyrepair(m):
    rft = _load(f'rft_functional_pyrepair_{m}.json')
    base = _load(f'functional_pyrepair_{ALT.get(m, m)}.json')
    out = {}
    if base:
        out['sft'] = base['sft_solved_rate'] * 100
        out['sf'] = base['synthfix_solved_rate'] * 100
    if rft:
        out['rft'] = rft['sft_solved_rate'] * 100
    return out


def codeflaws(m):
    rft = _load(f'rft_functional_codeflaws_{m}.json')
    bn = 'functional_codeflaws.json' if m == 'deepseek-1.3b' \
        else f'functional_codeflaws_{m}.json'
    base = _load(bn)
    out = {}
    if base:
        n = base['n_bugs']
        out['sft'] = base['metrics']['sft_greedy']['solved_count'] / n * 100
        out['sf'] = base['metrics']['synthfix_bestofk']['solved_count'] / n * 100
    if rft:
        n = rft['n_bugs']
        out['rft'] = rft['metrics']['sft_greedy']['solved_count'] / n * 100
    return out


def sven(m):
    rft = _load(f'rft_security_sven_{m}.json')
    base = _load('security_sven.json' if m == 'deepseek-1.3b'
                 else f'security_sven_{m}.json')
    out = {}
    if base:
        out['sft'] = base['security']['sft_cleared_rate'] * 100
        out['sf'] = base['security']['synthfix_cleared_rate'] * 100
    if rft:
        out['rft'] = rft['security']['sft_cleared_rate'] * 100
    return out


def codebleu(m):
    res = {}
    for ds in ('pyrepair', 'codeflaws', 'sven'):
        d = _load(f'matrix_rft/rft_{m}_{ds}.json')
        if d:
            cb = d['codebleu']
            res[ds] = cb * 100 if cb < 1.5 else cb
    return res


def fmt(d, k):
    return f'{d[k]:.1f}' if k in d else '  -- '


if __name__ == '__main__':
    print('=== Table 1 (functional pass@1 / security cleared, %) ===')
    print(f'{"model":14s} | pyrepair SFT/RFT/SF | codeflaws SFT/RFT/SF | '
          f'sven SFT/RFT/SF')
    for m in MODELS:
        p, c, s = pyrepair(m), codeflaws(m), sven(m)
        print(f'{m:14s} | '
              f'{fmt(p,"sft")} {fmt(p,"rft")} {fmt(p,"sf")} | '
              f'{fmt(c,"sft")} {fmt(c,"rft")} {fmt(c,"sf")} | '
              f'{fmt(s,"sft")} {fmt(s,"rft")} {fmt(s,"sf")}')
    print('\n=== Table 2 (RFT CodeBLEU, new budget-matched runs) ===')
    for m in MODELS:
        cb = codebleu(m)
        print(f'{m:14s} | ' + '  '.join(
            f'{ds}={cb.get(ds, float("nan")):.2f}' for ds in
            ('pyrepair', 'codeflaws', 'sven')))
