# SynthFix Artifact Results

This directory contains the JSON result files used by the paper tables.

## Protocol

All reported metrics are deployable repair metrics:

- **pyrepair / CodeFlaws:** functional pass@1, where a patch must compile and pass the held-out test suite.
- **SVEN:** security-cleared rate, where Semgrep no longer reports the target vulnerability.

Inference-time selection is leak-free: candidate ranking uses public tests and static symbolic features only; held-out tests are used only for reporting. SynthFix samples `K=16` candidates and applies a greedy floor.

## Main Results

### pyrepair (Python, n=115)

| Model | SFT greedy | RFT greedy | SynthFix best-of-K | Oracle@K |
|---|---:|---:|---:|---:|
| DeepSeek-1.3B | 68.7 | 69.6 | **87.0** | 90.4 |
| Llama-3.2-3B | 73.9 | 73.9 | **85.2** | 87.8 |
| Qwen3-4B-Base | 80.9 | 80.9 | **91.3** | 93.0 |
| CodeLLaMA-7B | 72.2 | 75.7 | **81.7** | 82.6 |
| StarCoder2-7B | 79.1 | 76.5 | **93.0** | 93.9 |

### CodeFlaws (C, n=389)

| Model | SFT greedy | RFT greedy | SynthFix best-of-K | Oracle@K |
|---|---:|---:|---:|---:|
| DeepSeek-1.3B | 12.3 | 10.5 | **15.2** | 19.3 |
| Llama-3.2-3B | 14.4 | 13.6 | **22.1** | 27.2 |
| Qwen3-4B-Base | 18.8 | 15.9 | **22.6** | 28.8 |
| CodeLLaMA-7B | 12.9 | 12.6 | **18.8** | 21.3 |
| StarCoder2-7B | 15.7 | 15.7 | **22.6** | 27.5 |

### SVEN (security, n=16)

| Model | SFT greedy | RFT greedy | SynthFix best-of-K | Oracle@K |
|---|---:|---:|---:|---:|
| DeepSeek-1.3B | 87.5 | 87.5 | **100.0** | 100.0 |
| Llama-3.2-3B | 100.0 | 87.5 | 100.0 | 100.0 |
| Qwen3-4B-Base | 87.5 | 87.5 | **93.8** | 100.0 |
| CodeLLaMA-7B | 93.8 | 93.8 | 93.8 | 93.8 |
| StarCoder2-7B | 87.5 | 81.2 | **100.0** | 100.0 |

## Selection Efficiency

Headroom captured is:

```text
(SynthFix best-of-K - greedy) / (Oracle@K - greedy)
```

SynthFix captures most pyrepair headroom (average 87.4%), substantial CodeFlaws headroom (average 53.5%), and most aggregate SVEN headroom where nonzero headroom remains (average 83.3%). Run:

```bash
python artifact/verify_results.py
```

to recompute the per-model values.

## Focused Component Ablation

The main ablation is a single-model component analysis on DeepSeek-1.3B / CodeFlaws. It removes or replaces major pipeline components rather than trying to rank individual reward weights.

| Variant | Selection rule | Solved@1 |
|---|---|---:|
| SFT-only baseline | Greedy | 45 (11.6%) |
| RFT-only baseline | Greedy | 41 (10.5%) |
| No router (fixed schedule) | Greedy | 46 (11.8%) |
| No router (fixed schedule) | Symbolic best-of-K | 64 (16.5%) |
| SynthFix without test-time selector | Greedy | 50 (12.9%) |
| SynthFix with unranked candidates | Random pick from K | 41 (10.5%) |
| SynthFix full pipeline | Symbolic best-of-K | 66 (17.0%) |
| Oracle@K upper bound | Held-out oracle | 86 (22.1%) |

This supports the paper's component claim: RFT-only training is not enough, the router improves the trained policy over a fixed schedule, random sampling alone is worse than greedy, and the full symbolic selector gives the best non-oracle result.

## Files

- `functional_pyrepair_*.json`, `functional_codeflaws*.json`, and `security_sven*.json`: deployable metrics and oracle ceilings.
- `rft_functional_*.json`, `rft_security_*.json`: budget-matched RFT-only baselines.
- `matrix/`, `matrix_rft/`: training-matrix diagnostic outputs.
- `ablation/`: focused router/reward ablation results.
