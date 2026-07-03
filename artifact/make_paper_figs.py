#!/usr/bin/env python
"""Generate publication figures for the SynthFix paper from real result JSONs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / 'artifact' / 'results' / 'artifact_prep'
FIGS = ROOT / 'artifact' / 'generated_figures'
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.linewidth': 0.7,
    'axes.grid': True,
    'grid.alpha': 0.28,
    'grid.linewidth': 0.4,
    'legend.frameon': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

MODELS = ['deepseek', 'llama3.2-3b', 'qwen3-4b', 'codellama-7b', 'starcoder2-7b']
# RFT result files key deepseek as 'deepseek-1.3b'
RFT_NAME = {'deepseek': 'deepseek-1.3b'}
LABELS = ['DeepSeek-1.3B', 'Llama-3.2-3B', 'Qwen3-4B', 'CodeLLaMA-7B',
          'StarCoder2-7B']
SFT_C = '#6aa84f'   # baseline / greedy (matches prior draft green)
RFT_C = '#e69138'   # fixed schedule (matches prior draft orange)
RFT_BAR = '#c0392b' # RFT baseline bars in auxiliary functionality plot
SF_C = '#1f4e79'    # full SynthFix
FIX_C = '#5b8bb5'


def pyrepair():
    sft, rft, sf = [], [], []
    for m in MODELS:
        d = json.load(open(R / f'functional_pyrepair_{m}.json'))
        n = d['n']
        sft.append(d['sft_greedy_solved'] / n * 100)
        sf.append(d['synthfix_bestK_solved'] / n * 100)
        r = json.load(open(R / f'rft_functional_pyrepair_{RFT_NAME.get(m, m)}.json'))
        rft.append(r['sft_solved_rate'] * 100)
    return sft, rft, sf


def codeflaws():
    sft, rft, sf = [], [], []
    for m in MODELS:
        f = R / ('functional_codeflaws.json' if m == 'deepseek'
                 else f'functional_codeflaws_{m}.json')
        d = json.load(open(f))
        n = d['n_bugs']
        sft.append(d['metrics']['sft_greedy']['solved_count'] / n * 100)
        sf.append(d['metrics']['synthfix_bestofk']['solved_count'] / n * 100)
        r = json.load(open(R / f'rft_functional_codeflaws_{RFT_NAME.get(m, m)}.json'))
        rft.append(r['metrics']['sft_greedy']['solved_count'] / r['n_bugs'] * 100)
    return sft, rft, sf


def sven():
    sft, rft, sf = [], [], []
    for m in MODELS:
        f = R / ('security_sven.json' if m == 'deepseek'
                 else f'security_sven_{m}.json')
        s = json.load(open(f))['security']
        sft.append(s['sft_cleared_rate'] * 100)
        sf.append(s['synthfix_cleared_rate'] * 100)
        r = json.load(open(R / f'rft_security_sven_{RFT_NAME.get(m, m)}.json'))
        rft.append(r['security']['sft_cleared_rate'] * 100)
    return sft, rft, sf


ORC_C = '#c9c9c9'


def pyrepair_headroom():
    sft, bk, orc = [], [], []
    for m in MODELS:
        d = json.load(open(R / f'functional_pyrepair_{m}.json'))
        n = d['n']
        sft.append(d['sft_greedy_solved'] / n * 100)
        bk.append(d['synthfix_bestK_solved'] / n * 100)
        orc.append(d['oracle_solved'] / n * 100)
    return sft, bk, orc


def codeflaws_headroom():
    sft, bk, orc = [], [], []
    for m in MODELS:
        f = R / ('functional_codeflaws.json' if m == 'deepseek'
                 else f'functional_codeflaws_{m}.json')
        d = json.load(open(f))
        n = d['n_bugs']
        mt = d['metrics']
        sft.append(mt['sft_greedy']['solved_count'] / n * 100)
        bk.append(mt['synthfix_bestofk']['solved_count'] / n * 100)
        orc.append(mt['synthfix_oracle']['solved_count'] / n * 100)
    return sft, bk, orc


def fig_main():
    """Selection-efficiency figure: how much of the oracle-reachable repair
    headroom SynthFix's symbolic best-of-K selection actually captures. This
    complements (does not duplicate) Table 1 by adding the oracle@K ceiling."""
    import numpy as np
    py = pyrepair_headroom()
    cf = codeflaws_headroom()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    x = np.arange(len(MODELS))
    w = 0.27
    panels = [(py, 'pyrepair (Python, $n{=}115$)'),
              (cf, 'CodeFlaws (C, $n{=}389$)')]
    for ax, ((st, bk, orc), title) in zip(axes, panels):
        ax.bar(x + w, orc, w, label='Oracle@$K$ (ceiling)', color=ORC_C,
               edgecolor='black', linewidth=0.4, hatch='////')
        b1 = ax.bar(x - w, st, w, label='SFT (greedy)', color=SFT_C,
                    edgecolor='black', linewidth=0.4)
        b2 = ax.bar(x, bk, w, label='SynthFix (best-of-$K$)',
                    color=SF_C, edgecolor='black', linewidth=0.4)
        for b in list(b1) + list(b2):
            ax.annotate(f'{b.get_height():.0f}',
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha='center', va='bottom', fontsize=5.4,
                        xytext=(0, 1), textcoords='offset points')
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS, fontsize=6.4, rotation=30, ha='right')
        ax.set_ylabel('Functional pass@1 (\\%)', fontsize=7.5)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylim(0, max(orc) * 1.22)
        ax.tick_params(labelsize=6.5)
        ax.grid(axis='y', linewidth=0.3, alpha=0.4)
    handles, labels = axes[0].get_legend_handles_labels()
    order = [1, 2, 0]  # SFT, SynthFix, Oracle
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               frameon=False, fontsize=7.5, loc='upper center',
               bbox_to_anchor=(0.5, 1.06), ncol=3, columnspacing=1.6,
               handletextpad=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIGS / 'selection_headroom.pdf'
    fig.savefig(out, bbox_inches='tight')
    print('wrote', out)


def _curve_style():
    """Shared line styling aligned with the prior draft's rq2 epoch curves."""
    return dict(linewidth=1.5, markersize=3.2, markeredgewidth=0.5)


COL_W = 3.4  # single-column width (inches)
PANEL_H = 1.45  # height for 1x2 horizontal single-column figures


def _style_axes(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=6.2)
    ax.grid(axis='y', linewidth=0.4, alpha=0.28)
    ax.title.set_fontsize(7.8)


def _plot_selection_panel(ax, data, title, sty, ymin=0):
    me = max(1, len(data['x']) // 5)
    ax.plot(data['x'], data['greedy'], marker='o', color=SFT_C,
            markevery=me, label='Greedy', **sty)
    ax.plot(data['x'], data['bestk'], marker='s', color=SF_C,
            markevery=me, label='Symbolic best-of-$K$', **sty)
    ax.plot(data['x'], data['oracle'], marker='^', color=ORC_C,
            linestyle='--', markevery=me, label='Oracle@$K$', **sty)
    ax.set_title(title)
    ax.set_xlabel('Held-out bugs', fontsize=6.5)
    _style_axes(ax)
    ax.set_ylim(ymin, None)
    ax.margins(x=0.03)


def fig_ablations():
    """Test-time selection as cumulative scaling curves (RQ2 figure)."""
    import numpy as np

    def pct(counts, ns):
        return np.array(counts, dtype=float) / np.array(ns, dtype=float) * 100

    def codeflaws_selection_curves():
        d = json.load(open(R / 'ablation' / 'eval_infersel_codeflaws.json'))
        n = d['n_bugs']
        mt = d['metrics']
        xs = [n]
        greedy = [mt['synthfix_greedy']['solved_count']]
        bestk = [mt['synthfix_bestofk']['solved_count']]
        oracle = [mt['synthfix_oracle']['solved_count']]
        x = np.array(xs)
        return {
            'x': x,
            'greedy': pct(greedy, x),
            'bestk': pct(bestk, x),
            'oracle': pct(oracle, x),
        }

    def pyrepair_selection_curves():
        d = json.load(open(R / 'ablation' / 'eval_rq2_norouter_pyrepair.json'))
        n = d['n']
        xs = [n]
        sft = [d['sft_greedy_solved']]
        sf = [d['synthfix_bestK_solved']]
        oracle = [d['oracle_solved']]
        x = np.array(xs)
        return {
            'x': x,
            'greedy': pct(sft, x),
            'bestk': pct(sf, x),
            'oracle': pct(oracle, x),
        }

    sty = _curve_style()
    cf = codeflaws_selection_curves()
    py = pyrepair_selection_curves()

    fig, axes = plt.subplots(1, 2, figsize=(COL_W, PANEL_H), sharey=False)
    _plot_selection_panel(axes[0], cf, '(a) CodeFlaws', sty, ymin=10)
    _plot_selection_panel(axes[1], py, '(b) pyrepair', sty, ymin=50)
    axes[0].set_ylabel('Cumulative pass@1 (%)', fontsize=6.5)
    axes[1].set_ylabel('')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02),
               ncol=3, fontsize=6.2, columnspacing=0.8, handletextpad=0.25)

    fig.subplots_adjust(left=0.10, right=0.99, top=0.78, bottom=0.22, wspace=0.28)
    out = FIGS / 'ablations.pdf'
    fig.savefig(out, bbox_inches='tight', pad_inches=0.02)
    print('wrote', out)


def fig_functionality():
    """Auxiliary functionality diagnostic (FixJS + CodeFlaws EM), styled like Figure 2."""
    import numpy as np

    js_cats = ['Net', 'Data', 'DOM', 'Async', 'Util']
    js = {
        'SFT': [38, 35, 31, 25, 23],
        'RFT': [20, 19, 24, 25, 8],
        'SynthFix': [43, 27, 45, 65, 31],
    }
    c_cats = ['Math', 'Sys', 'Str', 'I/O', 'Struct']
    c_lang = {
        'SFT': [24, 22, 16, 7, 0],
        'RFT': [33, 28, 32, 52, 14],
        'SynthFix': [36, 36, 42, 59, 14],
    }
    colors = {'SFT': SFT_C, 'RFT': RFT_BAR, 'SynthFix': SF_C}
    labels = ['SFT', 'RFT', 'SynthFix']

    fig, axes = plt.subplots(1, 2, figsize=(COL_W, PANEL_H), sharey=True)
    datasets = [(js_cats, js, '(a) FixJS'),
                (c_cats, c_lang, '(b) CodeFlaws')]
    x = np.arange(5)
    w = 0.22

    for ax, (cats, data, title) in zip(axes, datasets):
        for i, lab in enumerate(labels):
            off = (i - 1) * w
            ax.bar(x + off, data[lab], w, label=lab, color=colors[lab],
                   edgecolor='black', linewidth=0.3)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(cats, fontsize=6.0, rotation=0, ha='center')
        ymax = max(max(v) for v in data.values())
        ax.set_ylim(0, max(55, ymax * 1.08))
        _style_axes(ax)

    axes[0].set_ylabel('Exact Match (%)', fontsize=6.5)
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc='upper center', bbox_to_anchor=(0.5, 1.02),
               ncol=3, fontsize=6.2, columnspacing=0.8, handletextpad=0.25)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.78, bottom=0.24, wspace=0.28)
    out = FIGS / 'functionality.pdf'
    fig.savefig(out, bbox_inches='tight', pad_inches=0.02)
    print('wrote', out)


def fig_rq3():
    import numpy as np
    d = json.load(open(R / 'ablation' / 'eval_infersel_codeflaws.json'))
    m = d['metrics']
    n = d['n_bugs']
    keys = ['synthfix_greedy', 'synthfix_random', 'synthfix_bestofk',
            'synthfix_oracle']
    names = ['Greedy', 'Random\npick', 'Symbolic\nbest-of-K', 'Oracle@K']
    vals = [m[k]['solved_count'] / n * 100 for k in keys]
    cols = [SFT_C, '#d98880', SF_C, '#cccccc']
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    b = ax.bar(names, vals, color=cols, edgecolor='black', linewidth=0.4,
               width=0.66)
    for bar in b:
        ax.annotate(f'{bar.get_height():.1f}',
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha='center', va='bottom', fontsize=7,
                    xytext=(0, 1), textcoords='offset points')
    ax.set_ylabel('Functional pass@1 (\\%)', fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(0, max(vals) * 1.18)
    ax.tick_params(labelsize=7)
    ax.grid(axis='y', linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    out = FIGS / 'rq3_selection.pdf'
    fig.savefig(out, bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    fig_main()
    fig_ablations()
    fig_functionality()
