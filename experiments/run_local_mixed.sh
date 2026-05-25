#!/usr/bin/env bash
# =============================================================================
# run_local_mixed.sh -- Two-stream orchestration:
#   * Stream A on GPUs 0+1 : the 7 TP-compatible models (AWQ + w4a16),
#     each run sequentially with tensor_parallel_size=2.
#   * Stream B on GPU 2    : the 3 bnb-only models (no AWQ/w4a16
#     equivalent on HF), each run sequentially with tensor_parallel_size=1.
#
# Streams A and B run in parallel as background bash subshells. Wall
# clock is max(A, B); on caribou at --limit 100 the projection is
# ~12.5 h (A dominated by qwq-32b + the two 70Bs).
#
# Compared to the alternatives:
#   - All sequential TP=2 (PR #31)  : ~10 h but 3 bnb crash without our
#                                     auto-cap; even with cap, the
#                                     3 bnb run on TP=1 sequentially
#                                     stacking onto the AWQ critical
#                                     path -> ~19 h.
#   - All parallel TP=1             : ~6-7 h but the 32B/70B thinking
#                                     models on a single GPU are
#                                     unusable (qwq-32b -> ~12 h).
#   - This mixed orchestration       : streams overlap, neither GPU set
#                                     stalls on the other.
#
# The split: model -> stream is decided by _quantization_for in
# experiments/evaluate_local_vllm.py: bnb -> stream B, anything else
# -> stream A. If a model's id is moved between bnb and AWQ in
# experiments/evaluate_models.py, that automatically reshuffles the
# streams here.
#
# Defaults match the run_local_sequential_tp.sh contract:
#   --gpus     comma list, default "0,1,2"
#   --limit    instances per dataset, default 5
#   --skip     comma list of model names to drop from both streams
#   --models   restrict to a comma list (then split into streams)
#
# Usage:
#   ./experiments/run_local_mixed.sh                              # default 3 GPUs, limit=5
#   ./experiments/run_local_mixed.sh --gpus 0,1,2 --limit 100
#   ./experiments/run_local_mixed.sh --skip qwq-32b               # skip qwq for a fast dry-run
#
# Logs (one file per model):
#   experiments/results/smoke_logs/mixed_<TS>_<model>.log
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

GPUS="0,1,2"
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
        -h|--help) sed -n '2,50p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

IFS=',' read -ra GPU_LIST <<< "$GPUS"
N_GPU=${#GPU_LIST[@]}
if [[ $N_GPU -lt 2 ]]; then
    echo "[ABORT] run_local_mixed needs at least 2 GPUs (one for stream A TP=2," \
         "one for stream B). Got $N_GPU: $GPUS" >&2
    exit 1
fi

# Stream A: TP-capable models -> use (N_GPU - 1) GPUs at TP=(N_GPU-1)
# Stream B: bnb-only models   -> the last GPU at TP=1
STREAM_A_GPUS=$(IFS=','; echo "${GPU_LIST[*]:0:$((N_GPU - 1))}")
STREAM_A_TP=$((N_GPU - 1))
STREAM_B_GPU="${GPU_LIST[$((N_GPU - 1))]}"

# Lists must match what _quantization_for in evaluate_local_vllm.py returns.
# Stream A: anything not "bitsandbytes" (AWQ + w4a16 + FP8 + GPTQ + MXFP4).
# Order: heavy first so we get partial-data fallback if a SIGINT happens
# late.
STREAM_A_MODELS=(
    llama-3.3-70b
    deepseek-r1-distill-70b
    qwen2.5-72b
    gpt-oss-120b
    qwq-32b
    qwen3-30b-a3b
    gemma-3-27b
    gpt-oss-20b
    mistral-small-3
    gemma-3-12b
    qwen3-14b
    phi-4
)
# Stream B: small model that stays single-GPU. granite-3.2-8b is the
# only model below 12 B and runs in FP16 (~16 GB), so NCCL overhead of
# TP=2 outweighs the gain at this size. Solo on GPU 2 while stream A
# handles the heavier models on GPUs 0+1.
STREAM_B_MODELS=(
    granite-3.2-8b
)

# Apply --models filter (only keep names that appear in the selection)
if [[ -n "$SELECTED_MODELS" ]]; then
    FILTERED_A=()
    FILTERED_B=()
    for m in "${STREAM_A_MODELS[@]}"; do
        [[ ",$SELECTED_MODELS," =~ ,$m, ]] && FILTERED_A+=("$m")
    done
    for m in "${STREAM_B_MODELS[@]}"; do
        [[ ",$SELECTED_MODELS," =~ ,$m, ]] && FILTERED_B+=("$m")
    done
    STREAM_A_MODELS=("${FILTERED_A[@]}")
    STREAM_B_MODELS=("${FILTERED_B[@]}")
fi

# Apply --skip filter
if [[ -n "$SKIP_MODELS" ]]; then
    FILTERED_A=()
    FILTERED_B=()
    for m in "${STREAM_A_MODELS[@]}"; do
        [[ ! ",$SKIP_MODELS," =~ ,$m, ]] && FILTERED_A+=("$m")
    done
    for m in "${STREAM_B_MODELS[@]}"; do
        [[ ! ",$SKIP_MODELS," =~ ,$m, ]] && FILTERED_B+=("$m")
    done
    STREAM_A_MODELS=("${FILTERED_A[@]}")
    STREAM_B_MODELS=("${FILTERED_B[@]}")
fi

# Datasets (skip private ones unless their files exist)
DATASETS=(financebench finqa tatqa medcalc medqa headqa cuad maud contractnli)
[[ -f "$PROJECT_DIR/dataset/all_manual_evaluations.jsonl" ]] && DATASETS+=(rag_insurance)
[[ -f "$PROJECT_DIR/dataset/insurance_text_simplifications_annotated.jsonl" ]] && DATASETS+=(judgebert)

echo "============================================================"
echo " LOCAL MIXED RUN  (stream A parallel with stream B)"
echo "============================================================"
echo "  Stream A   : GPUs $STREAM_A_GPUS (TP=$STREAM_A_TP) -- ${STREAM_A_MODELS[*]:-(empty)}"
echo "  Stream B   : GPU  $STREAM_B_GPU (TP=1)         -- ${STREAM_B_MODELS[*]:-(empty)}"
echo "  Limit      : $LIMIT instances per dataset"
echo "  Datasets   : ${DATASETS[*]} (n=${#DATASETS[@]})"
echo "============================================================"
echo ""

LOG_DIR="$PROJECT_DIR/experiments/results/smoke_logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

run_model() {
    # $1 model name, $2 gpu list, $3 tp size
    local model="$1"
    local gpu="$2"
    local tp="$3"
    local log="$LOG_DIR/mixed_${TS}_${model}.log"
    echo "  [$(date +%H:%M:%S)] $model on GPU $gpu (TP=$tp) -> $log"
    for ds in "${DATASETS[@]}"; do
        local t0
        t0=$(date +%s)
        if stdbuf -oL -eL env PYTHONUNBUFFERED=1 \
                PYTHONPATH=src python3 -u -m experiments.evaluate_models \
                --dataset "$ds" --model "$model" \
                --limit "$LIMIT" --gpu "$gpu" \
                --tensor-parallel-size "$tp" --force \
                >> "$log" 2>&1; then
            echo "  [$(date +%H:%M:%S)] $model x $ds OK ($(( $(date +%s) - t0 ))s)"
        else
            echo "  [$(date +%H:%M:%S)] $model x $ds FAIL ($(( $(date +%s) - t0 ))s) -- see $log"
        fi
    done
}

stream_A() {
    echo "[stream A] start (${#STREAM_A_MODELS[@]} models on GPUs $STREAM_A_GPUS TP=$STREAM_A_TP)"
    for m in "${STREAM_A_MODELS[@]}"; do
        run_model "$m" "$STREAM_A_GPUS" "$STREAM_A_TP"
    done
    echo "[stream A] DONE"
}

stream_B() {
    echo "[stream B] start (${#STREAM_B_MODELS[@]} models on GPU $STREAM_B_GPU TP=1)"
    for m in "${STREAM_B_MODELS[@]}"; do
        run_model "$m" "$STREAM_B_GPU" "1"
    done
    echo "[stream B] DONE"
}

START_ALL=$(date +%s)

# Launch both streams as bash subshells in the background so they run
# concurrently. wait blocks until both finish.
stream_A &
PID_A=$!
stream_B &
PID_B=$!

wait "$PID_A"
RC_A=$?
wait "$PID_B"
RC_B=$?

END_ALL=$(date +%s)
ELAPSED=$(( END_ALL - START_ALL ))

echo ""
echo "============================================================"
echo " MIXED RUN DONE"
echo "============================================================"
echo "  Stream A exit : $RC_A"
echo "  Stream B exit : $RC_B"
echo "  Wall time     : $((ELAPSED / 60))m $((ELAPSED % 60))s"
echo "  Logs          : $LOG_DIR/mixed_${TS}_*.log"
echo "============================================================"

# Exit non-zero if either stream had a non-zero exit
exit $(( RC_A + RC_B ))
