#!/usr/bin/env bash
# =============================================================================
# run_local_large.sh — Large local models (27B+) on a single GPU
#
# Models: QwQ-32B, Qwen3-30B-A3B, Gemma-2-27B
# Default GPU: 0
#
# Usage:
#   ./experiments/run_local_large.sh                    # GPU 0 (wandb on)
#   ./experiments/run_local_large.sh --gpu 1            # GPU 1
#   ./experiments/run_local_large.sh --limit 50
#   ./experiments/run_local_large.sh --force            # Overwrite all existing results
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

MODELS=(qwq-32b qwen3-30b-a3b gemma-2-27b)
GPU="0"
EXTRA_ARGS=()

# Extract --gpu from args, pass everything else through
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu) GPU="$2"; shift 2 ;;
        *)     EXTRA_ARGS+=("$1"); shift ;;
    esac
done

echo "=== Local LARGE models (GPU $GPU) ==="
echo "  Models: ${MODELS[*]}"
echo ""

for model in "${MODELS[@]}"; do
    exec_args=(python -m experiments.evaluate_models
        --dataset all --model "$model" --gpu "$GPU"
        "${EXTRA_ARGS[@]}")
    echo "[RUN] ${exec_args[*]}"
    "${exec_args[@]}" || echo "[FAIL] $model (continuing...)"
done
