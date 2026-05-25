#!/usr/bin/env bash
# =============================================================================
# run_thinking_tp.sh -- Phase 2 : run thinking models sequentially with
# tensor parallelism across all available GPUs.
#
# Why a separate script:
#   - vLLM tensor_parallel_size=N shards one model across N GPUs, so we
#     can only run ONE thinking model at a time but each model is ~N x
#     faster than on a single GPU. With 3 x RTX 6000 Ada (48 GB), the
#     32-70B thinking models that would take 12 h alone now run in ~4-5 h.
#   - Putting this in run_local_smoke_parallel.sh would conflate two very
#     different orchestration shapes (multi-bucket parallel vs. single TP).
#
# Models targeted (matches _THINKING_MODEL_PATTERNS in evaluate_models.py):
#     qwen3-14b, qwen3-30b-a3b, qwq-32b, deepseek-r1-distill-70b
#
# Usage:
#   # Phase 1 first (non-thinking models, ~7 h on 3 GPUs):
#   ./experiments/run_local_smoke_parallel.sh --gpus 0,1,2 --limit 100 --skip-thinking
#
#   # Then Phase 2 (thinking models with TP=3, ~10 h sequentially):
#   ./experiments/run_thinking_tp.sh --gpus 0,1,2 --limit 100
#
#   # Skip a specific thinking model:
#   ./experiments/run_thinking_tp.sh --gpus 0,1,2 --limit 100 --skip qwen3-14b
#
# Datasets are the same as run_local_smoke.sh.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

GPUS="0,1,2"
LIMIT="5"
SKIP_MODELS=""

_require_arg() {
    if [[ $# -lt 2 || -z "${2:-}" || "${2:0:2}" == "--" ]]; then
        echo "[ABORT] $1 requires a value" >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)   _require_arg "$@"; GPUS="$2"; shift 2 ;;
        --limit)  _require_arg "$@"; LIMIT="$2"; shift 2 ;;
        --skip)   _require_arg "$@"; SKIP_MODELS="$2"; shift 2 ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

IFS=',' read -ra GPU_LIST <<< "$GPUS"
N_GPU=${#GPU_LIST[@]}
if [[ $N_GPU -lt 1 ]]; then
    echo "[ABORT] no GPUs selected"; exit 1
fi

# Thinking model order: cheapest first so we get usable data early if
# the long ones get killed.
ALL_THINKING=(qwen3-14b qwen3-30b-a3b qwq-32b deepseek-r1-distill-70b)

# Apply --skip
MODELS=()
for m in "${ALL_THINKING[@]}"; do
    if [[ ! ",$SKIP_MODELS," =~ ,$m, ]]; then
        MODELS+=("$m")
    fi
done

DATASETS=(financebench finqa tatqa medcalc medqa headqa cuad maud contractnli)
[[ -f "$PROJECT_DIR/dataset/all_manual_evaluations.jsonl" ]] && DATASETS+=(rag_insurance)
[[ -f "$PROJECT_DIR/dataset/insurance_text_simplifications_annotated.jsonl" ]] && DATASETS+=(judgebert)

echo "============================================================"
echo " THINKING MODELS -- TENSOR PARALLEL"
echo "============================================================"
echo "  GPUs           : $GPUS  (n=$N_GPU, tensor_parallel_size=$N_GPU)"
echo "  Limit          : $LIMIT instances per dataset"
echo "  Skip           : ${SKIP_MODELS:-(none)}"
echo "  Thinking queue : ${MODELS[*]:-(empty)}"
echo "  Datasets       : ${DATASETS[*]} (n=${#DATASETS[@]})"
echo "============================================================"
echo ""

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "[ABORT] no thinking models to run (all skipped?)"
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
    log="$LOG_DIR/thinking_tp_${TS}_${model}.log"
    echo "[$(date +%H:%M:%S)] >>> $model (log: $log)"

    for ds in "${DATASETS[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo "  [$CURRENT/$TOTAL] $model x $ds (limit=$LIMIT, tp=$N_GPU)"
        T0=$(date +%s)
        # stdbuf for line-buffered subprocess output (matches the
        # parallel-smoke fix in run_local_smoke.sh / .._parallel.sh).
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
echo " THINKING TP DONE"
echo "============================================================"
echo "  Total    : $TOTAL"
echo "  Passed   : $PASSED"
echo "  Failed   : $FAILED"
echo "  Time     : $((ELAPSED / 60))m $((ELAPSED % 60))s"
echo "  Logs     : $LOG_DIR/thinking_tp_${TS}_*.log"
echo "============================================================"

exit "$FAILED"
