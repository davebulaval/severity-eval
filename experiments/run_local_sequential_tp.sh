#!/usr/bin/env bash
# =============================================================================
# run_local_sequential_tp.sh -- Run every local model sequentially with
# vLLM tensor parallelism across all available GPUs.
#
# Rationale:
#   - Phase-1-parallel + Phase-2-TP felt nice on paper but the GPU2
#     bucket on phase 1 (gemma-2-27b alone) dominated wall clock.
#   - Routing every model through TP=N collapses both phases into one
#     sequential pass. Each model gets all GPUs at once, so:
#       * the 70B AWQ models keep enough KV cache to load CUAD at 33 K
#         (no truncation, no per-model max_model_len cap on llama-3.3)
#       * the bottleneck (qwq-32b thinking, gemma-2-27b non-thinking)
#         shrinks roughly 2-2.5x vs single-GPU
#       * total wall at limit=100 drops from ~16.5 h to ~10 h
#   - Smaller models (granite-3.2-8b, qwen3-14b) pay an NCCL overhead
#     under TP=3 but the loss is a few minutes and avoids juggling two
#     orchestration shapes.
#
# Defaults: every local model in MODELS (evaluate_models.py),
# every dataset (private datasets included if their files exist).
#
# Usage:
#   ./experiments/run_local_sequential_tp.sh                       # default 3 GPUs, --limit 5
#   ./experiments/run_local_sequential_tp.sh --gpus 0,1,2 --limit 100
#   ./experiments/run_local_sequential_tp.sh --gpus 0 --limit 10   # single GPU (TP=1)
#   ./experiments/run_local_sequential_tp.sh --skip qwq-32b,deepseek-r1-distill-70b
#   ./experiments/run_local_sequential_tp.sh --models phi-4,granite-3.2-8b
#
# Each (model, dataset) result lands at experiments/results/<dataset>_<model>.json.
# Per-model log: experiments/results/smoke_logs/sequential_tp_<TS>_<model>.log
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Ordered cheap -> heavy so a SIGINT mid-run still gives us results on
# the small models. Matches MODELS in evaluate_models.py.
DEFAULT_MODELS=(
    granite-3.2-8b
    gemma-2-9b
    qwen3-14b
    phi-4
    mistral-small-3
    gemma-2-27b
    qwen3-30b-a3b
    qwq-32b
    llama-3.3-70b
    deepseek-r1-distill-70b
)

# Default to 2 GPUs because vLLM requires
#   num_attention_heads % tensor_parallel_size == 0
# and every local model we run has num_heads in {16, 32, 40, 64} --
# none of which is divisible by 3. TP=3 fails fast with
#   "Total number of attention heads (N) must be divisible by tensor
#   parallel size (3)"
# from VllmConfig validation. TP=2 covers all four head counts (16/2=8,
# 32/2=16, 40/2=20, 64/2=32). The third card stays idle on a 3-GPU
# machine; use --gpus 0,1,2,3 once a 4th GPU is available.
GPUS="0,1"
LIMIT="5"
SKIP_MODELS=""
SELECTED_MODELS=""

_require_arg() {
    if [[ $# -lt 2 || -z "${2:-}" || "${2:0:2}" == "--" ]]; then
        echo "[ABORT] $1 requires a value" >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)    _require_arg "$@"; GPUS="$2"; shift 2 ;;
        --limit)   _require_arg "$@"; LIMIT="$2"; shift 2 ;;
        --skip)    _require_arg "$@"; SKIP_MODELS="$2"; shift 2 ;;
        --models)  _require_arg "$@"; SELECTED_MODELS="$2"; shift 2 ;;
        -h|--help) sed -n '2,35p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

IFS=',' read -ra GPU_LIST <<< "$GPUS"
N_GPU=${#GPU_LIST[@]}
if [[ $N_GPU -lt 1 ]]; then
    echo "[ABORT] no GPUs selected"; exit 1
fi

# Build model list
if [[ -n "$SELECTED_MODELS" ]]; then
    IFS=',' read -ra MODELS <<< "$SELECTED_MODELS"
else
    MODELS=("${DEFAULT_MODELS[@]}")
fi
if [[ -n "$SKIP_MODELS" ]]; then
    FILTERED=()
    for m in "${MODELS[@]}"; do
        if [[ ! ",$SKIP_MODELS," =~ ,$m, ]]; then
            FILTERED+=("$m")
        fi
    done
    MODELS=("${FILTERED[@]}")
fi

# Datasets to evaluate (skip private ones unless their files exist)
DATASETS=(
    financebench
    finqa
    tatqa
    medcalc
    medqa
    headqa
    cuad
    maud
    contractnli
)
[[ -f "$PROJECT_DIR/dataset/all_manual_evaluations.jsonl" ]] && DATASETS+=(rag_insurance)
[[ -f "$PROJECT_DIR/dataset/insurance_text_simplifications_annotated.jsonl" ]] && DATASETS+=(judgebert)

echo "============================================================"
echo " LOCAL SEQUENTIAL -- TENSOR PARALLEL"
echo "============================================================"
echo "  GPUs           : $GPUS  (n=$N_GPU, tensor_parallel_size=$N_GPU)"
echo "  Limit          : $LIMIT instances per dataset"
echo "  Skip           : ${SKIP_MODELS:-(none)}"
echo "  Models queue   : ${MODELS[*]:-(empty)}"
echo "  Datasets       : ${DATASETS[*]} (n=${#DATASETS[@]})"
echo "============================================================"
echo ""

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "[ABORT] no models to run (all skipped?)"
    exit 1
fi

LOG_DIR="$PROJECT_DIR/experiments/results/smoke_logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

START_ALL=$(date +%s)
PASSED=0
FAILED=0
TOTAL=$(( ${#MODELS[@]} * ${#DATASETS[@]} ))
CURRENT=0

for model in "${MODELS[@]}"; do
    log="$LOG_DIR/sequential_tp_${TS}_${model}.log"
    echo "[$(date +%H:%M:%S)] >>> $model (log: $log)"

    for ds in "${DATASETS[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo "  [$CURRENT/$TOTAL] $model x $ds (limit=$LIMIT, tp=$N_GPU)"
        T0=$(date +%s)
        # stdbuf line-buffers child output so `tail -f` reflects progress
        # in real time. python3 -u + PYTHONUNBUFFERED=1 handle Python's
        # block-buffered stdout when the destination is a regular file.
        if stdbuf -oL -eL env PYTHONUNBUFFERED=1 \
                PYTHONPATH=src python3 -u -m experiments.evaluate_models \
                --dataset "$ds" --model "$model" \
                --limit "$LIMIT" --gpu "$GPUS" \
                --tensor-parallel-size "$N_GPU" --force \
                >> "$log" 2>&1; then
            T1=$(date +%s)
            echo "  [$CURRENT/$TOTAL] OK in $((T1 - T0))s"
            PASSED=$((PASSED + 1))
        else
            T1=$(date +%s)
            echo "  [$CURRENT/$TOTAL] FAIL in $((T1 - T0))s -- see $log"
            FAILED=$((FAILED + 1))
        fi
    done
done

END_ALL=$(date +%s)
ELAPSED=$(( END_ALL - START_ALL ))

echo ""
echo "============================================================"
echo " SEQUENTIAL TP DONE"
echo "============================================================"
echo "  Total    : $TOTAL"
echo "  Passed   : $PASSED"
echo "  Failed   : $FAILED"
echo "  Time     : $((ELAPSED / 60))m $((ELAPSED % 60))s"
echo "  Logs     : $LOG_DIR/sequential_tp_${TS}_*.log"
echo "============================================================"

exit "$FAILED"
