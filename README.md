# SynthFix Artifact

This repository contains the artifact for **SynthFix: Adaptive Neuro-Symbolic Code Vulnerability Repair**.

SynthFix is an adaptive neuro-symbolic repair pipeline. During training, a router chooses between supervised fine-tuning (SFT) and reward fine-tuning (RFT). The reward uses compiler-, analyzer-, and test-derived symbolic evidence. At inference time, the same evidence ranks a small candidate set with a greedy floor, so the selected patch is chosen by deployable signals rather than token likelihood alone.

## What Is Included

```text
synthfix-artifact/
├── README.md
├── LICENSE
├── data/
│   └── README.md
├── paper/
│   └── acl_latex.pdf
└── artifact/
    ├── src/
    │   ├── train_synthfix.py
    │   └── train_baseline.py
    ├── run_all_experiments.py
    ├── queue_artifact.py
    ├── run_eval_sweep.py
    ├── run_rft_matrix.py
    ├── run_ablations.py
    ├── collect_rft.py
    ├── codebleu_one.py
    ├── eval_functional.py
    ├── eval_functional_py.py
    ├── eval_security_sven.py
    ├── make_paper_figs.py
    ├── verify_results.py
    ├── requirements.txt
    └── results/artifact_prep/
        ├── ARTIFACT_RESULTS.md
        ├── functional_*.json
        ├── security_sven*.json
        ├── rft_*.json
        ├── matrix/
        ├── matrix_rft/
        └── ablation/
```

Generated LoRA checkpoints, benchmark corpora, caches, and verbose run logs are not bundled to keep the repository lightweight. The scripts expect checkpoints under `artifact/checkpoints/` and benchmark data under `data/` or `artifact/work/` when rerunning training/evaluation. The packaged JSON files under `artifact/results/artifact_prep/` are included because they support the paper-facing verification script.

## Setup

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r artifact/requirements.txt
```

For Qwen3-4B-Base, use a newer isolated environment with compatible `torch` and `transformers` versions, as noted in `artifact/queue_artifact.py`.

## Verify Packaged Results

The packaged JSON files reproduce the paper-facing result tables:

```bash
python artifact/verify_results.py
```

This prints:

- Table 1-style main deployable metrics for pyrepair, CodeFlaws, and SVEN.
- Table 3-style oracle headroom and captured fraction.
- Table 5-style component ablation values.

## Hugging Face Models

SynthFix LoRA repair-agent adapters and their matching router checkpoints are available on Hugging Face:

- `CoderDoge/synthfix-deepseek-coder-1.3b-pyrepair`
- `CoderDoge/synthfix-deepseek-coder-1.3b-sven`
- `CoderDoge/synthfix-starcoder2-7b-codeflaws`
- `CoderDoge/synthfix-starcoder2-7b-pyrepair`
- `CoderDoge/synthfix-starcoder2-7b-sven`
- `CoderDoge/synthfix-codellama-7b-codeflaws`
- `CoderDoge/synthfix-codellama-7b-pyrepair`
- `CoderDoge/synthfix-codellama-7b-sven`

Each model repository contains the PEFT/LoRA adapter files, tokenizer metadata, and `router.pt`. The adapters should be loaded with the corresponding base model specified in the model card.

## Reproduce Experiments

Build pyrepair:

```bash
python artifact/build_pyexec_benchmark.py
```

Run the full training matrix:

```bash
python artifact/queue_artifact.py
```

Run evaluation sweeps:

```bash
python artifact/run_eval_sweep.py
```

Run focused ablations:

```bash
python artifact/run_ablations.py
```

For GPU runs, pin the visible device and pass `--gpu 0`, for example:

```bash
CUDA_VISIBLE_DEVICES=0 python artifact/run_all_experiments.py --worker \
  --method synthfix --model_name deepseek-1.3b --dataset_name codeflaws \
  --data_dir artifact/work/codeflaws_exec --gpu 0 \
  --epochs 4 --sft_warmup_epochs 2 --batch_size 16 --lr 2e-4 \
  --rl_beta 0.12 --kl_beta 0.12 --rloo_k 2 --use_rich_exec_reward \
  --select_metric val_codebleu \
  --save_ckpt_to artifact/checkpoints/matrix/synthfix_deepseek-1.3b_codeflaws \
  --out artifact/results/artifact_prep/matrix/synthfix_deepseek-1.3b_codeflaws.json
```

## Main Claims Supported By This Artifact

- SynthFix improves deployable repair metrics over SFT-only and RFT-only baselines across pyrepair, CodeFlaws, and SVEN.
- Symbolic best-of-`K` selection recovers substantial oracle-reachable headroom while preserving a greedy floor.
- The focused component ablation on DeepSeek-1.3B/CodeFlaws shows that RFT-only training is not enough, adaptive routing improves the trained policy over a fixed schedule, random sampling alone is worse than greedy, and the full symbolic selector gives the best non-oracle result.

## License

MIT. See `LICENSE`.
