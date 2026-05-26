#!/usr/bin/env bash
# =============================================================================
# run_local_mixed.sh -- Two-stream orchestration on 3 GPUs:
#   * Stream A on GPUs 0+1 (TP=2) : the 5 models that genuinely benefit
#     from or require tensor parallelism (gpt-oss-120b needs TP=2 to
#     fit; the three 70-72 B FP8 checkpoints + qwq-32b get a real KV
#     cache lift at TP=2).
#   * Stream B on GPU 2 (TP=1)    : the 7 models that fit comfortably
#     on one 48 GB card (qwen3-30b-a3b, gemma-3-{12,27}b, qwen3-14b,
#     gpt-oss-20b, phi-4, granite-3.2-8b). Each is ~30-50% slower at
#     TP=1 than at TP=2, but Stream B used to sit on a single model
#     (granite) and idle the rest of the run -- moving the smaller
#     models there shaves ~30% off wall time at --limit 1000.
#
# Streams A and B run in parallel as background bash subshells. Wall
# clock is max(A, B); on caribou the projection at --limit 1000 (8
# datasets, 12 models) is ~4.5 days vs ~6.4 days with the previous
# single-model Stream B.
#
# The lists are hard-coded above based on per-model size and KV-cache
# behaviour; if you add a model to MODELS, place it in Stream A if it
# is >36 GB FP8 or thinking-class, otherwise Stream B. mistral-small-3
# is deliberately omitted (broken tokenizer on the RedHat FP8
# checkpoint -- emits raw BPE tokens).
#
# Defaults match the run_local_sequential_tp.sh contract:
#   --gpus      comma list, default "0,1,2"
#   --limit     instances per dataset, default 5
#   --skip      comma list of model names to drop from both streams
#   --models    restrict to a comma list (then split into streams)
#   --datasets  restrict to a comma list (subset of the retained 7+1)
#   --force     pass --force to evaluate_models so existing output JSONs
#               are rebuilt from scratch (instead of being extended via
#               the PR #40 resume-by-id path)
#
# Usage:
#   ./experiments/run_local_mixed.sh                              # default 3 GPUs, limit=5
#   ./experiments/run_local_mixed.sh --gpus 0,1,2 --limit 100
#   ./experiments/run_local_mixed.sh --skip qwq-32b               # skip qwq for a fast dry-run
#   ./experiments/run_local_mixed.sh --datasets headqa,maud --force --limit 1000
#                                                                 # targeted re-run after the
#                                                                 # PR #50 unique-id fix
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
SELECTED_DATASETS=""
FORCE=""

_require_arg() {
    if [[ $# -lt 2 || -z "${2:-}" || "${2:0:2}" == "--" ]]; then
        echo "[ABORT] $1 requires a value" >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)     _require_arg "$@"; GPUS="$2"; shift 2 ;;
        --limit)    _require_arg "$@"; LIMIT="$2"; shift 2 ;;
        --skip)     _require_arg "$@"; SKIP_MODELS="$2"; shift 2 ;;
        --models)   _require_arg "$@"; SELECTED_MODELS="$2"; shift 2 ;;
        --datasets) _require_arg "$@"; SELECTED_DATASETS="$2"; shift 2 ;;
        --force)    FORCE="--force"; shift ;;
        -h|--help)  sed -n '2,50p' "$0"; exit 0 ;;
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

# Split is by *physical requirement*, not size, to maximise 3-GPU usage:
#  * Stream A (TP=2, GPUs 0+1) gets the models that genuinely benefit from
#    or require tensor parallelism -- gpt-oss-120b at 63 GB needs TP=2 to
#    fit at all, the three 70-72B FP8 checkpoints get a substantial KV
#    cache lift at TP=2, and qwq-32b's thinking budget pushes the same
#    way.
#  * Stream B (TP=1, GPU 2) gets everything that fits comfortably on a
#    single 48 GB card. Running these single-GPU costs ~30-50% per-prompt
#    throughput vs TP=2, but freeing the GPU-2-idle slot saves more wall
#    time than the per-model slowdown costs. Order: heaviest first so a
#    late SIGINT keeps the small-model JSONs intact.
STREAM_A_MODELS=(
    deepseek-r1-distill-70b
    qwq-32b
    qwen2.5-72b
    llama-3.3-70b
    gpt-oss-120b
)
STREAM_B_MODELS=(
    qwen3-30b-a3b
    gemma-3-27b
    qwen3-14b
    gemma-3-12b
    gpt-oss-20b
    phi-4
    granite-3.2-8b
)
# mistral-small-3 deliberately omitted from both lists: its
# RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic checkpoint
# emits raw BPE tokens (e.g. "Ġ;)ĊĊĠterzo...") instead of decoded text,
# so accuracy is 0% across every dataset in the smoke run. Pass it via
# --models if you want to re-validate after a checkpoint swap.

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

# Datasets that survived the "domain-expert-derived vs heuristic-only"
# audit. medqa, medcalc, and rag_insurance were dropped because their
# severity label is assigned by keyword matching on raw question text
# -- too noisy to defend in the paper. What remains is 7 datasets:
#   * 4 domain-expert-derived (cuad, maud, contractnli, headqa) where
#     severity comes from a structured categorical field (clause type,
#     deal-point category, fixed hypothesis, academic category);
#   * 3 finance datasets (financebench, finqa, tatqa) where severity
#     comes from a matrix over structured answer/program/metric axes
#     -- still heuristic, but defensible and disclosed in Limitations.
DATASETS=(financebench finqa tatqa headqa cuad maud contractnli)
[[ -f "$PROJECT_DIR/dataset/insurance_text_simplifications_annotated.jsonl" ]] && DATASETS+=(judgebert)

# Apply --datasets filter (only keep names that appear in the selection).
# Useful for targeted re-runs, e.g. `--datasets headqa,maud --force` after
# the unique-id loader fix to rebuild only those two datasets.
if [[ -n "$SELECTED_DATASETS" ]]; then
    FILTERED_DS=()
    for d in "${DATASETS[@]}"; do
        [[ ",$SELECTED_DATASETS," == *",$d,"* ]] && FILTERED_DS+=("$d")
    done
    if [[ ${#FILTERED_DS[@]} -eq 0 ]]; then
        echo "[ABORT] --datasets '$SELECTED_DATASETS' did not match any of:" \
             "${DATASETS[*]}" >&2
        exit 1
    fi
    DATASETS=("${FILTERED_DS[@]}")
fi

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
        # $FORCE is "--force" when the script was invoked with --force,
        # empty otherwise. With --force the per-(model, dataset) output
        # JSON is rebuilt from scratch; without it, the resume / extend
        # path in evaluate_local_vllm (PR #40) keeps existing rows and
        # only runs new ids.
        if stdbuf -oL -eL env PYTHONUNBUFFERED=1 \
                PYTHONPATH=src python3 -u -m experiments.evaluate_models \
                --dataset "$ds" --model "$model" \
                --limit "$LIMIT" --gpu "$gpu" \
                --tensor-parallel-size "$tp" $FORCE \
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
