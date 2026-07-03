#!/usr/bin/env python
"""Verify paper-facing SynthFix result tables from packaged JSON files."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
R = ROOT / "results" / "artifact_prep"

MODELS = [
    ("deepseek", "DeepSeek-1.3B", "deepseek-1.3b"),
    ("llama3.2-3b", "Llama-3.2-3B", "llama3.2-3b"),
    ("qwen3-4b", "Qwen3-4B-Base", "qwen3-4b"),
    ("codellama-7b", "CodeLLaMA-7B", "codellama-7b"),
    ("starcoder2-7b", "StarCoder2-7B", "starcoder2-7b"),
]


def load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def pct(count: int, total: int) -> float:
    return 100.0 * count / total


def captured(greedy: float, best: float, oracle: float) -> str:
    denom = oracle - greedy
    if abs(denom) < 1e-9:
        return "--"
    return f"{(best - greedy) / denom * 100:.1f}%"


def pyrepair(model_key: str) -> tuple[float, float, float]:
    d = load(R / f"functional_pyrepair_{model_key}.json")
    n = d["n"]
    return (
        pct(d["sft_greedy_solved"], n),
        pct(d["synthfix_bestK_solved"], n),
        pct(d["oracle_solved"], n),
    )


def codeflaws(model_key: str) -> tuple[float, float, float]:
    path = R / ("functional_codeflaws.json" if model_key == "deepseek" else f"functional_codeflaws_{model_key}.json")
    d = load(path)
    n = d["n_bugs"]
    m = d["metrics"]
    return (
        pct(m["sft_greedy"]["solved_count"], n),
        pct(m["synthfix_bestofk"]["solved_count"], n),
        pct(m["synthfix_oracle"]["solved_count"], n),
    )


def sven(model_key: str) -> tuple[float, float, float]:
    path = R / ("security_sven.json" if model_key == "deepseek" else f"security_sven_{model_key}.json")
    s = load(path)["security"]
    return (
        s["sft_cleared_rate"] * 100,
        s["synthfix_cleared_rate"] * 100,
        s["oracle_cleared_rate"] * 100,
    )


def print_main_and_headroom() -> None:
    print("\n== Main deployable metrics and headroom ==")
    print("Model\tpyrepair SFT/SynthFix/Oracle/Captured\tCodeFlaws SFT/SynthFix/Oracle/Captured\tSVEN SFT/SynthFix/Oracle/Captured")
    for model_key, label, _ in MODELS:
        py = pyrepair(model_key)
        cf = codeflaws(model_key)
        sv = sven(model_key)
        print(
            f"{label}\t"
            f"{py[0]:.1f}/{py[1]:.1f}/{py[2]:.1f}/{captured(*py)}\t"
            f"{cf[0]:.1f}/{cf[1]:.1f}/{cf[2]:.1f}/{captured(*cf)}\t"
            f"{sv[0]:.1f}/{sv[1]:.1f}/{sv[2]:.1f}/{captured(*sv)}"
        )


def print_rft_baselines() -> None:
    print("\n== RFT greedy baselines ==")
    for model_key, label, rft_key in MODELS:
        py = load(R / f"rft_functional_pyrepair_{rft_key}.json")
        cf = load(R / f"rft_functional_codeflaws_{rft_key}.json")
        sv = load(R / f"rft_security_sven_{rft_key}.json")
        cf_n = cf["n_bugs"]
        cf_solved = cf["metrics"]["sft_greedy"]["solved_count"]
        print(
            f"{label}: pyrepair={py['sft_solved_rate'] * 100:.1f}, "
            f"CodeFlaws={pct(cf_solved, cf_n):.1f}, "
            f"SVEN={sv['security']['sft_cleared_rate'] * 100:.1f}"
        )


def print_ablations() -> None:
    print("\n== Component ablation: DeepSeek-1.3B / CodeFlaws ==")
    full = load(R / "ablation" / "eval_infersel_codeflaws.json")
    norouter = load(R / "ablation" / "eval_rq2_norouter_codeflaws.json")
    rft = load(R / "rft_functional_codeflaws_deepseek-1.3b.json")
    n = full["n_bugs"]
    fm = full["metrics"]
    nm = norouter["metrics"]
    rm = rft["metrics"]
    rows = [
        ("SFT-only baseline", "Greedy", fm["sft_greedy"]["solved_count"]),
        ("RFT-only baseline", "Greedy", rm["sft_greedy"]["solved_count"]),
        ("No router (fixed schedule)", "Greedy", nm["synthfix_greedy"]["solved_count"]),
        ("No router (fixed schedule)", "Symbolic best-of-K", nm["synthfix_bestofk"]["solved_count"]),
        ("SynthFix without test-time selector", "Greedy", fm["synthfix_greedy"]["solved_count"]),
        ("SynthFix with unranked candidates", "Random pick from K", fm["synthfix_random"]["solved_count"]),
        ("SynthFix full pipeline", "Symbolic best-of-K", fm["synthfix_bestofk"]["solved_count"]),
        ("Oracle@K upper bound", "Held-out oracle", fm["synthfix_oracle"]["solved_count"]),
    ]
    for variant, rule, count in rows:
        print(f"{variant}: {rule}: {count} ({pct(count,n):.1f}%)")


def main() -> None:
    print_main_and_headroom()
    print_rft_baselines()
    print_ablations()


if __name__ == "__main__":
    main()
