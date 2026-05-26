#!/usr/bin/env bash
# =============================================================================
# run_local_3way.sh -- 3 parallel TP=1 streams, one model per GPU at a time.
#
# Maximises 3-GPU usage for re-run / extension passes where every model
# fits on a single 48 GB card (true for all retained FP8 / MXFP4
# checkpoints in the current MODELS table). All three GPUs work
# simultaneously instead of the Stream A (GPUs 0+1 TP=2) + Stream B
# (GPU 2 TP=1) split of run_local_mixed.sh which idles GPU 2 once
# Stream B's smallest queue empties.
#
# Models are partitioned across the GPUs in a 4-4-3 (or similar) split
# so wall time is min(max-load-imbalance). The current split is tuned
# for the headqa + maud re-run after the PR #50 / #51 / #52 fixes.
#
# Optional:
#   --datasets x,y      restrict to a comma list of datasets
#   --models a,b,c      restrict to a comma list of models (skip
#                       anything not on the list)
#   --skip a,b,c        comma list of models to drop
#   --limit N           instances per dataset (default 1000)
#   --force             rebuild JSONs from scratch (default: extend)
#
# Usage:
#   ./experiments/run_local_3way.sh --datasets headqa,maud --force --limit 1000
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

LIMIT="1000"
SELECTED_DATASETS=""
SELECTED_MODELS=""
SKIP_MODELS=""
FORCE=""

_require_arg() {
    if [[ $# -lt 2 || -z "${2:-}" || "${2:0:2}" == "--" ]]; then
        echo "[ABORT] $1 requires a value" >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --datasets) _require_arg "$@"; SELECTED_DATASETS="$2"; shift 2 ;;
        --models)   _require_arg "$@"; SELECTED_MODELS="$2"; shift 2 ;;
        --skip)     _require_arg "$@"; SKIP_MODELS="$2"; shift 2 ;;
        --limit)    _require_arg "$@"; LIMIT="$2"; shift 2 ;;
        --force)    FORCE="--force"; shift ;;
        -h|--help)  sed -n '2,32p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Heaviest first per GPU, sized so the slowest single model on each
# stream finishes around the same time. deepseek-r1-distill-70b is the
# heaviest (thinking + 70 B), so it gets its own GPU. qwq-32b and
# qwen2.5-72b are second tier. Smaller models bundle up on GPU 2.
# Tweak as needed: the script only requires that every listed model
# actually exists in MODELS and fits a single 48 GB card.
GPU0_MODELS=(deepseek-r1-distill-70b qwen3-30b-a3b gemma-3-12b)
GPU1_MODELS=(qwq-32b qwen2.5-72b qwen3-14b phi-4)
GPU2_MODELS=(gemma-3-27b gpt-oss-20b granite-3.2-8b)

_filter() {
    # Apply --skip and --models filters in place on $1 (array name).
    local -n arr=$1
    local out=()
    for m in "${arr[@]}"; do
        # Apply --models inclusion if non-empty
        if [[ -n "$SELECTED_MODELS" ]]; then
            [[ ",$SELECTED_MODELS," == *",$m,"* ]] || continue
        fi
        # Apply --skip exclusion
        if [[ -n "$SKIP_MODELS" && ",$SKIP_MODELS," == *",$m,"* ]]; then
            continue
        fi
        out+=("$m")
    done
    arr=("${out[@]}")
}
_filter GPU0_MODELS
_filter GPU1_MODELS
_filter GPU2_MODELS

LOG_DIR="$PROJECT_DIR/experiments/results/smoke_logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
ROOT_LOG="$LOG_DIR/run_local_3way_${TS}.log"

echo "============================================================"
echo " run_local_3way   $(date +%Y-%m-%d\ %H:%M:%S)"
echo "============================================================"
echo "  GPU 0 : ${GPU0_MODELS[*]:-(empty)}"
echo "  GPU 1 : ${GPU1_MODELS[*]:-(empty)}"
echo "  GPU 2 : ${GPU2_MODELS[*]:-(empty)}"
echo "  Datasets : ${SELECTED_DATASETS:-(all)}"
echo "  Limit    : $LIMIT"
echo "  Force    : ${FORCE:-(no)}"
echo "  Root log : $ROOT_LOG"
echo "============================================================"

_launch_gpu() {
    local gpu="$1"
    shift
    local models="$1"
    if [[ -z "$models" ]]; then
        echo "[GPU $gpu] empty model list, skipping"
        return 0
    fi
    bash "$SCRIPT_DIR/run_per_gpu.sh" \
        --gpu "$gpu" --models "$models" \
        --limit "$LIMIT" \
        ${SELECTED_DATASETS:+--datasets "$SELECTED_DATASETS"} \
        $FORCE
}

START=$(date +%s)

_launch_gpu 0 "$(IFS=','; echo "${GPU0_MODELS[*]}")" >> "$ROOT_LOG" 2>&1 &
PID0=$!
_launch_gpu 1 "$(IFS=','; echo "${GPU1_MODELS[*]}")" >> "$ROOT_LOG" 2>&1 &
PID1=$!
_launch_gpu 2 "$(IFS=','; echo "${GPU2_MODELS[*]}")" >> "$ROOT_LOG" 2>&1 &
PID2=$!

wait $PID0; RC0=$?
wait $PID1; RC1=$?
wait $PID2; RC2=$?

END=$(date +%s)
ELAPSED=$((END - START))

echo ""
echo "============================================================"
echo " 3-WAY RUN DONE"
echo "============================================================"
echo "  GPU 0 exit : $RC0"
echo "  GPU 1 exit : $RC1"
echo "  GPU 2 exit : $RC2"
echo "  Wall time  : $((ELAPSED / 60))m $((ELAPSED % 60))s"
echo "  Logs       : $LOG_DIR/pergpu_${TS}_*.log"
echo "============================================================"

exit $(( RC0 + RC1 + RC2 ))
