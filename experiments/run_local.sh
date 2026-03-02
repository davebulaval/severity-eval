#!/usr/bin/env bash
# =============================================================================
# run_local.sh — Run all local models in parallel on 2 GPUs
#
# Large models (27B+) on GPU 0, small/medium models on GPU 1.
# Override GPUs with --gpu-large and --gpu-small.
#
# Usage:
#   ./experiments/run_local.sh                          # GPU 0 + GPU 1 (wandb on)
#   ./experiments/run_local.sh --gpu-large 2 --gpu-small 3
#   ./experiments/run_local.sh --limit 50
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

GPU_LARGE="0"
GPU_SMALL="1"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu-large) GPU_LARGE="$2"; shift 2 ;;
        --gpu-small) GPU_SMALL="$2"; shift 2 ;;
        *)           EXTRA_ARGS+=("$1"); shift ;;
    esac
done

echo "=== Launching local models on 2 GPUs ==="
echo "  Large (27B+): GPU $GPU_LARGE"
echo "  Small (<27B): GPU $GPU_SMALL"
echo ""

bash "$SCRIPT_DIR/run_local_large.sh" --gpu "$GPU_LARGE" "${EXTRA_ARGS[@]}" &
PID_LARGE=$!

bash "$SCRIPT_DIR/run_local_small.sh" --gpu "$GPU_SMALL" "${EXTRA_ARGS[@]}" &
PID_SMALL=$!

# Wait for both and propagate failures
FAILED=0
wait $PID_LARGE || FAILED=$((FAILED + 1))
wait $PID_SMALL || FAILED=$((FAILED + 1))

echo ""
if [[ $FAILED -eq 0 ]]; then
    echo "=== All local models completed ==="
else
    echo "=== $FAILED GPU group(s) had failures ==="
fi

exit $FAILED
