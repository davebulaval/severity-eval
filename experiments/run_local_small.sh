#!/usr/bin/env bash
# =============================================================================
# run_local_small.sh — Small/medium local models (<27B) on a single GPU
#
# Models: Qwen3-14B, DeepSeek-R1-Distill-14B, Gemma-2-9B, Granite-3.2-8B
# Default GPU: 1
#
# Usage:
#   ./experiments/run_local_small.sh                    # GPU 1 (wandb on)
#   ./experiments/run_local_small.sh --gpu 0            # GPU 0
#   ./experiments/run_local_small.sh --limit 50
#   ./experiments/run_local_small.sh --force            # Overwrite all existing results
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

MODELS=(qwen3-14b deepseek-r1-distill-14b gemma-2-9b granite-3.2-8b)
GPU="1"
EXTRA_ARGS=()

# Extract --gpu from args, pass everything else through
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu) GPU="$2"; shift 2 ;;
        *)     EXTRA_ARGS+=("$1"); shift ;;
    esac
done

echo "=== Local SMALL models (GPU $GPU) ==="
echo "  Models: ${MODELS[*]}"
echo ""

for model in "${MODELS[@]}"; do
    exec_args=(python -m experiments.evaluate_models
        --dataset all --model "$model" --gpu "$GPU"
        "${EXTRA_ARGS[@]}")
    echo "[RUN] ${exec_args[*]}"
    "${exec_args[@]}" || echo "[FAIL] $model (continuing...)"
done
