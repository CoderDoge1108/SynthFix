# SynthFix

**Adaptive neural-symbolic code vulnerability repair for code LLMs.**

SynthFix is a PyTorch / Hugging Face training library that adds an
adaptive curriculum on top of standard supervised fine-tuning for
code-repair models. A small neural **router** inspects each training
example's difficulty features and decides, per sample, how strongly a
**symbolic reward** signal should shape the gradient. Easy samples are
learned with plain cross-entropy; harder samples additionally receive
a variance-reduced REINFORCE update driven by a composite symbolic
reward that scores generations along syntactic, control-flow,
security, and surface-similarity dimensions.

The library ships with:

- A drop-in training loop (`src/train_synthfix.py`) that works with
  any decoder-only LM exposed through `AutoModelForCausalLM`.
- SFT and RFT baseline trainers that share the same data loader.
- A split symbolic reward that doubles as a feature extractor for an
  inference-time best-of-K reranker.
- A data-processing utility for three common code-repair benchmarks
  (FixJS, CodeFlaws, SVEN) plus a unified JSON schema you can emit
  from your own data.
- Orchestrators that run the full training matrix end-to-end and
  aggregate per-run JSONs into a single report.

---

## Method

The training pipeline has three phases:

| Phase | What happens |
|-------|--------------|
| 1. SFT warmup | Standard cross-entropy fine-tuning on `(buggy → fixed)` pairs. Per-sample loss is recorded as a difficulty signal. |
| 2. Router pre-training | A 2-layer MLP router is supervised to predict "above-median loss" from four features: AST complexity, CFG depth, code length, and the current per-sample loss. |
| 3. Router-gated RFT | For each batch: (a) compute the SFT anchor loss, (b) draw `K` on-policy continuations (RLOO baseline), (c) score each with a split symbolic reward, (d) gate the RL contribution by the router's probability that a sample is "hard", (e) combine as `L = L_SFT + β · gate · A_LOO · CE_sampled`. |

### Split symbolic reward

Every generated continuation is scored on four dimensions in `[0, 1]`
(see `src/models/symbolic.py` and `src/models/reward.py`):

- `r_AST` — syntactic correctness via bracket balance.
- `r_CFG` — control-flow fidelity versus the reference, computed as
  normalized LCS over control-flow keyword sequences.
- `r_Sem` — security heuristic that penalizes common vulnerability
  patterns (`eval`, `exec`, `strcpy`, `innerHTML =`, `pickle.loads`,
  `shell=True`, …).
- `r_SIM` — surface similarity to the reference fix via character
  n-gram F-score (chrF, local implementation).

The composite reward is `r = λ_AST·r_AST + λ_CFG·r_CFG + λ_Sem·r_Sem +
λ_SIM·r_SIM`, with the same components exposed as a dictionary so
they can be reused as features for an inference-time reranker
(`src/models/inference.py`).

### Router

Minimal MLP:

```
[f_AST, f_CFG, f_len, f_loss]  →  64 (ReLU)  →  64 (ReLU)  →  1 (σ)  →  P(hard)
```

Features are min-max normalized per batch. The router is trained in
two stages: a short supervised pre-training step that predicts
"above-median loss" after the SFT warmup, plus an online BCE update
each batch in phase 3 using the current empirical median as the
target.

### Router gating

During phase 3, the RL term of the loss is multiplied by
`gate = 1[P(hard) ≥ 0.5]`. Easy samples (gate = 0) receive only the
SFT anchor; hard samples (gate = 1) additionally receive the
variance-reduced REINFORCE signal from RLOO. Advantages are
normalized to unit standard deviation across the batch and clamped
to the non-negative half-line, so updates only amplify rollouts that
beat their leave-one-out baseline.

### Design rationale

- **Why gate RFT?** Unfiltered RFT spends its noisy reward signal on
  easy samples that SFT already handles, adding variance without
  improving the policy. Gating concentrates the reward on the hard
  tail, where it helps most.
- **Why split the reward?** Collapsing the reward to a single scalar
  makes it easy for a single component (typically chrF) to dominate.
  Exposing the four components separately gives a richer training
  signal and lets the inference-time reranker score candidates
  multi-objectively.
- **Why RLOO with `K = 2`?** It is the minimal setting that still
  admits an unbiased, variance-reduced REINFORCE estimator (one
  sample acts as the baseline for the other), and it keeps the extra
  generation cost per step bounded.
- **Why keep an SFT anchor in phase 3?** The anchor keeps the policy
  close to the teacher distribution and prevents the collapse modes
  that pure policy-gradient fine-tuning is prone to on small
  repair datasets.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.9+ and a CUDA-capable GPU are recommended. LoRA fine-tuning
works comfortably on a single 24 GB card for 7B models and on a
single 12 GB card for 1.3B / 3B configurations.

---

## Data

`src/data/process_benchmarks.py` converts three common code-repair
benchmarks into a unified JSON format:

- **FixJS** (JavaScript) —
  `fixjs/input/{50, 50-100, 100+}/{before, after}_tokenized.txt`
- **CodeFlaws** (C) —
  `codeflaws/<id>-bug-<bugID>-<fixID>/*.c`
- **SVEN** (Python) —
  `sven/data_train_val/{train, val}/cwe-*.jsonl`

Each processed dataset is written as `{train,val,test}.json`, where
every record has the shape
`{"buggy": str, "fixed": str, "language": str}`. An 80/10/10 split is
produced with `seed=13`.

Point the orchestrator at your raw data via the `SYNTHFIX_DATA`
environment variable (parent directory containing `fixjs/`,
`codeflaws/`, and `sven/`) or by editing the `RAW_DATA` default in
`run_all_experiments.py`. Benchmark data is not bundled with this
repository; obtain it from the upstream sources.

### Using your own data

The loader in `src/data/dataset.py` consumes the unified JSON schema
directly. Any dataset whose records can be expressed as
`{"buggy": str, "fixed": str, "language": str}` will work without
touching the training code — just emit `train.json`, `val.json`, and
`test.json` in a directory and pass that directory as `--dataset`.

---

## Usage

### Train on a single (model, dataset) pair

```bash
python -m src.train_synthfix \
    --model deepseek-1.3b \
    --dataset data/processed/fixjs \
    --output runs/synthfix_deepseek-1.3b_fixjs \
    --epochs 4 --batch_size 16 --lr 2e-4
```

### Train an SFT or RFT baseline

```bash
python -m src.train_baseline \
    --method sft \
    --model deepseek-1.3b \
    --dataset data/processed/fixjs \
    --output runs/sft_deepseek-1.3b_fixjs
```

### Two-stage training (warm-start SynthFix from a trained SFT checkpoint)

```bash
python orchestrate_twostage.py
```

### Full matrix driver

```bash
python run_all_experiments.py --seed 42
```

### End-to-end pipeline

```bash
bash run_final_pipeline.sh
```

Per-run JSONs emitted by the orchestrators can be rolled up into a
single report with `aggregate_final.py`.

---

## Models

The training code is written against `AutoModelForCausalLM`, so any
decoder-only code LM can be dropped in by extending the `MODEL_PATHS`
registry in `src/train_synthfix.py`. The registry shipped here covers
a representative range of open-source code models:

| Model          |  Size | Hugging Face ID                         |
|----------------|------:|------------------------------------------|
| DeepSeek-Coder |  1.3B | `deepseek-ai/deepseek-coder-1.3b-base`   |
| StarCoder2     |    3B | `bigcode/starcoder2-3b`                  |
| CodeLlama      |    7B | `codellama/CodeLlama-7b-hf`              |
| DeepSeek-Coder |  6.7B | `deepseek-ai/deepseek-coder-6.7b-base`   |
| StarCoder2     |    7B | `bigcode/starcoder2-7b`                  |

Parameter-efficient fine-tuning uses LoRA (rank = 16, alpha = 32,
dropout = 0.05) on `{q, k, v, o}_proj` by default.

---

## Extending the framework

Common extensions and where to make them:

- **New reward components.** Add a scoring function to
  `src/models/symbolic.py`, include its output in
  `compute_reward_split`, and mix it into
  `compute_reward_from_split` with a new weight. The inference
  reranker picks up the component automatically.
- **Different router features.** `compute_batch_features` in
  `src/models/router.py` is the single place to extend the feature
  vector; also update the `input_size` passed to `RouterModel(...)`.
- **Different gating policy.** The hard-sample gate is a single
  expression in `src/train_synthfix.py`
  (`hard_gate = (prob_hard >= 0.5).float()`). Swap in a soft gate
  (`hard_gate = prob_hard`) or a temperature-scaled variant without
  touching anything else.
- **New benchmark.** Add a `process_<name>(raw_dir, output_base)`
  function to `src/data/process_benchmarks.py` that emits the same
  JSON schema, then point the orchestrator at the new dataset name.

---

## Repository layout

```
SynthFix/
├── run_all_experiments.py      # Full (model × dataset × method) matrix driver
├── orchestrate_final.py        # End-to-end SFT → SynthFix → aggregate pipeline
├── orchestrate_twostage.py     # Two-stage (SFT warm-start) driver
├── aggregate_final.py          # Collect per-run JSONs into a single report
├── run_final_pipeline.sh       # One-shot pipeline entry point
├── diag_synthfix_eval.py       # Standalone per-run diagnostics
├── diag_ensemble_eval.py       # SFT × SynthFix ensemble diagnostics
├── requirements.txt
├── LICENSE
├── configs/
│   ├── deepspeed_zero2.json
│   └── deepspeed_zero3.json
├── scripts/
│   └── run_sensitivity.sh      # Reward-weight sweep
├── src/
│   ├── train_synthfix.py       # Router-gated REINFORCE training loop
│   ├── train_baseline.py       # SFT / RFT baselines (shared data path)
│   ├── data/
│   │   ├── dataset.py          # RepairDataset + causal-LM collate
│   │   └── process_benchmarks.py  # Raw → unified JSON splits
│   └── models/
│       ├── router.py           # MLP router + feature extraction
│       ├── reward.py           # Composite symbolic reward
│       ├── symbolic.py         # Split symbolic features (train + inference)
│       └── inference.py        # Best-of-K reranker
└── data/                       # Place processed benchmark splits here
```

---

## Contributing

Issues and pull requests are welcome. If you find a bug, please open
an issue with a minimal reproduction. For larger changes, opening an
issue first to discuss the direction is appreciated.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{synthfix2026,
  title     = {SynthFix: Adaptive Neural-Symbolic Code Vulnerability Repair},
  author    = {SynthFix Authors},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  year      = {2026},
}
```

## License

Released under the terms of the [LICENSE](LICENSE) file.
