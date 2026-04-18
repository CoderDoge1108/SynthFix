#!/usr/bin/env bash
# ==============================================================================
# Reward Weight Sensitivity Analysis
# ==============================================================================
# Tests different reward weight configurations:
#   - Equal:    λ_AST=0.333, λ_CFG=0.333, λ_Sem=0.334
#   - AST-heavy: λ_AST=0.5, λ_CFG=0.25, λ_Sem=0.25
#   - CFG-heavy: λ_AST=0.25, λ_CFG=0.5, λ_Sem=0.25
#   - Sem-heavy: λ_AST=0.25, λ_CFG=0.25, λ_Sem=0.5
# ==============================================================================
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/src"

GPU=${1:-0}
MODEL="codellama-7b"
DATASET="fixjs"
DATA_DIR="../data/processed/${DATASET}"
EPOCHS=10
BATCH_SIZE=16

echo "============================================================"
echo "  Reward Weight Sensitivity Analysis"
echo "  Model: ${MODEL} | Dataset: ${DATASET}"
echo "============================================================"

declare -A CONFIGS
CONFIGS["equal"]="0.333 0.333 0.334"
CONFIGS["ast_heavy"]="0.5 0.25 0.25"
CONFIGS["cfg_heavy"]="0.25 0.5 0.25"
CONFIGS["sem_heavy"]="0.25 0.25 0.5"

for CONFIG_NAME in "${!CONFIGS[@]}"; do
    read -r AST CFG SEM <<< "${CONFIGS[$CONFIG_NAME]}"

    echo ""
    echo "[${CONFIG_NAME}] λ_AST=${AST}, λ_CFG=${CFG}, λ_Sem=${SEM}"

    python train_synthfix.py \
        --model "$MODEL" \
        --dataset "$DATA_DIR" \
        --output "../outputs/sensitivity/${DATASET}_${MODEL}_${CONFIG_NAME}" \
        --epochs $EPOCHS --batch_size $BATCH_SIZE --gpu $GPU \
        --lambda_ast "$AST" --lambda_cfg "$CFG" --lambda_sem "$SEM"

    echo "✓ ${CONFIG_NAME} complete"
done

echo ""
echo "============================================================"
echo "  Sensitivity Analysis Complete — Results in outputs/sensitivity/"
echo "============================================================"
