# SynthFix: Adaptive Neural-Symbolic Code Vulnerability Repair

> **Status — preview artifact.** This repository accompanies the paper
> *"SynthFix: Adaptive Neural-Symbolic Code Vulnerability Repair."*
> The full experiment suite is still running at the time of this release,
> so **numerical results and trained model checkpoints are deliberately
> omitted and marked as TBD** in the tables below. The complete method,
> training code, evaluation pipeline, and reproducibility scripts are all
> included so reviewers can re-run the pipeline end-to-end. We will push
> the final tables and release the checkpoints in a tagged update before
> the camera-ready deadline.

SynthFix is a hybrid training framework for LLM-based code vulnerability
repair. It combines Supervised Fine-Tuning (SFT) with a Reward
Fine-Tuning (RFT) signal through a **lightweight neural router** that
decides, per sample, how strongly the symbolic reward should guide
training. The router implements an adaptive curriculum: model capacity
is focused on harder repair patterns while easy samples are learned via
plain SFT.

---

## 1. Method

The training pipeline has three phases:

| Phase | Description |
|-------|-------------|
| 1. SFT warmup (Epoch 1) | Pure cross-entropy fine-tuning on `(buggy → fixed)` pairs. Per-sample loss is recorded as a difficulty signal. |
| 2. Router pre-training | A 2-layer MLP router is supervised to predict "above-median loss" from four features: AST complexity, CFG depth, code length, and current per-sample loss. |
| 3. Router-gated RFT (Epochs 2..E) | For each batch: (a) compute SFT anchor loss, (b) draw `K=2` on-policy samples (RLOO baseline), (c) score each with a split symbolic reward `r = λ_AST·r_AST + λ_CFG·r_CFG + λ_Sem·r_Sem + λ_SIM·r_SIM`, (d) gate the RL contribution by the router's "hard-sample" probability, (e) combine as `L = L_SFT + β · gate · A_LOO · CE_sampled`. |

**Split symbolic reward** (in `src/models/symbolic.py`). Every generated
continuation is scored on four dimensions in `[0,1]`:

* `r_AST` — syntactic correctness via bracket balance.
* `r_CFG` — control-flow fidelity vs. reference (LCS on keyword sequences).
* `r_Sem` — security heuristic (vulnerability pattern penalty).
* `r_SIM` — character n-gram F-score (chrF) vs. reference fix.

**Router** (in `src/models/router.py`). MLP with architecture
`[4] → 64 (ReLU) → 64 (ReLU) → 1 (sigmoid)`. Output is interpreted as
`P(hard sample)`. Features are min-max normalized per batch.

**Why it works.** Unfiltered RFT spends its noisy reward signal on easy
samples that SFT already handles, adding variance without improving
the policy. Router gating concentrates the reward on the hard tail,
where it measurably helps. Splitting the reward into AST / CFG / SEM /
SIM components (rather than collapsing to a single scalar) provides a
richer training signal and is also reused as features for the
inference-time reranker (`src/models/inference.py`).

A formal version of the algorithm, gating analysis, and a LoRA-level
capacity argument appear in Sections 3.1-3.3 of the paper.

---

## 2. Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, CUDA-capable GPU (24 GB VRAM recommended
for 7B models; 12 GB sufficient for the 1.3B / 3B configurations with
LoRA).

---

## 3. Data

SynthFix is evaluated on three vulnerability-repair benchmarks:

| Dataset    | Language   | Train  | Val  | Test |
|------------|------------|-------:|-----:|-----:|
| FixJS      | JavaScript |  2,000 |  100 |  200 |
| CodeFlaws  | C          | ~3,100 | ~390 | ~390 |
| SVEN       | Python     |   ~716 |  ~42 |  ~42 |

Place raw benchmark data under `../data/raw_benchmarks/` (relative to
the repo root), **or** export `SYNTHFIX_DATA` to a directory containing
`fixjs/`, `codeflaws/`, `sven/` subfolders. The experiment runner will
process and split them automatically using `seed=13` (80/10/10).

The expected raw layouts are documented in
`src/data/process_benchmarks.py`:

* FixJS:     `fixjs/input/{50,50-100,100+}/{before,after}_tokenized.txt`
* CodeFlaws: `codeflaws/<id>-bug-<bugID>-<fixID>/*.c`
* SVEN:      `sven/data_train_val/{train,val}/cwe-*.jsonl`

---

## 4. Running Experiments

### Full suite (5 models × 3 datasets × {SFT, RFT, SynthFix})

```bash
python run_all_experiments.py --seed 42
```

Outputs under `results/` (ignored by git).

### Single (model, dataset) pair

```bash
python -m src.train_synthfix \
    --model deepseek-1.3b \
    --dataset data/fixjs \
    --output results/deepseek-1.3b_fixjs \
    --epochs 4 --batch_size 16 --lr 2e-4
```

### Baselines (SFT or RFT only)

```bash
python -m src.train_baseline \
    --method sft --model deepseek-1.3b \
    --dataset data/fixjs --output results/sft_deepseek-1.3b_fixjs
```

### Two-stage SynthFix (warm-start from a pretrained SFT checkpoint)

```bash
python orchestrate_twostage.py --model deepseek-1.3b --dataset data/fixjs
```

### Final reproducibility pipeline (SFT → SynthFix + aggregation)

```bash
bash run_final_pipeline.sh
```

---

## 5. Models

| Model            | Size | HuggingFace ID                           |
|------------------|-----:|------------------------------------------|
| DeepSeek-Coder   | 1.3B | `deepseek-ai/deepseek-coder-1.3b-base`   |
| StarCoder2       |   3B | `bigcode/starcoder2-3b`                  |
| CodeLlama        |   7B | `codellama/CodeLlama-7b-hf`              |
| DeepSeek-Coder   | 6.7B | `deepseek-ai/deepseek-coder-6.7b-base`   |
| StarCoder2       |   7B | `bigcode/starcoder2-7b`                  |

All models use LoRA (rank=16, alpha=32, dropout=0.05) on
`{q,k,v,o}_proj` for parameter-efficient fine-tuning.

---

## 6. Results (to be released)

> The numbers in this section are **placeholders** and will be replaced
> with the final values in a tagged update before the camera-ready
> deadline. Metrics reported: **CodeBLEU** (structural code similarity)
> and **Exact Match (EM)** (fraction of predictions identical to the
> reference fix).

### 6.1 Main comparison — CodeBLEU (%)

| Model           | Method    | FixJS | CodeFlaws | SVEN |
|-----------------|-----------|------:|----------:|-----:|
| DeepSeek-1.3B   | SFT       |   TBD |       TBD |  TBD |
| DeepSeek-1.3B   | RFT       |   TBD |       TBD |  TBD |
| DeepSeek-1.3B   | SynthFix  |   TBD |       TBD |  TBD |
| StarCoder2-3B   | SFT       |   TBD |       TBD |  TBD |
| StarCoder2-3B   | RFT       |   TBD |       TBD |  TBD |
| StarCoder2-3B   | SynthFix  |   TBD |       TBD |  TBD |
| CodeLlama-7B    | SFT       |   TBD |       TBD |  TBD |
| CodeLlama-7B    | RFT       |   TBD |       TBD |  TBD |
| CodeLlama-7B    | SynthFix  |   TBD |       TBD |  TBD |
| DeepSeek-6.7B   | SFT       |   TBD |       TBD |  TBD |
| DeepSeek-6.7B   | RFT       |   TBD |       TBD |  TBD |
| DeepSeek-6.7B   | SynthFix  |   TBD |       TBD |  TBD |
| StarCoder2-7B   | SFT       |   TBD |       TBD |  TBD |
| StarCoder2-7B   | RFT       |   TBD |       TBD |  TBD |
| StarCoder2-7B   | SynthFix  |   TBD |       TBD |  TBD |

### 6.2 Exact Match (%)

Layout identical to §6.1; values TBD.

### 6.3 Ablations

| Ablation                                          | FixJS | CodeFlaws | SVEN |
|---------------------------------------------------|------:|----------:|-----:|
| Full SynthFix                                     |   TBD |       TBD |  TBD |
| − router gate (always-on RFT)                     |   TBD |       TBD |  TBD |
| − split reward (single scalar)                    |   TBD |       TBD |  TBD |
| − RLOO (single-sample baseline)                   |   TBD |       TBD |  TBD |
| − SFT anchor                                      |   TBD |       TBD |  TBD |
| − loss-history feature for router                 |   TBD |       TBD |  TBD |

### 6.4 Reward-weight sensitivity (CodeBLEU)

See `scripts/run_sensitivity.sh`. Final heatmap will be added in the
camera-ready revision.

---

## 7. Project Structure

```
SynthFix/
├── run_all_experiments.py      # Full matrix orchestrator (5 models × 3 datasets)
├── orchestrate_final.py        # Final SFT → SynthFix → aggregate pipeline
├── orchestrate_twostage.py     # Two-stage (SFT warm-start) driver
├── aggregate_final.py          # Collect per-run JSONs into the final report
├── run_final_pipeline.sh       # One-shot reproducibility entry point
├── requirements.txt
├── LICENSE
├── configs/
│   ├── deepspeed_zero2.json
│   └── deepspeed_zero3.json
├── scripts/
│   └── run_sensitivity.sh      # λ_AST/λ_CFG/λ_Sem/λ_SIM sweep
├── src/
│   ├── train_synthfix.py       # SynthFix v11 training loop (router-gated RFT)
│   ├── train_baseline.py       # SFT / RFT baselines (identical data loader)
│   ├── data/
│   │   ├── dataset.py          # RepairDataset + causal-LM collate
│   │   └── process_benchmarks.py  # Raw → unified JSON splits
│   └── models/
│       ├── router.py           # MLP router + feature extraction
│       ├── reward.py           # Composite symbolic reward (training)
│       ├── symbolic.py         # Split symbolic features (train + inference)
│       └── inference.py        # Best-of-K reranker using split-symbolic features
├── tests/
│   └── test_method_comparison.py
└── diag_synthfix_eval.py       # Standalone per-run diagnostics
```

---

## 8. Citation

A BibTeX entry will be added with the camera-ready version. Please cite
the paper if you use this code.

## 9. License

Released under the terms of the [LICENSE](LICENSE) file.
