#!/usr/bin/env bash
# =============================================================================
# run_per_gpu.sh -- single-GPU sequential runner.
#
# Designed to be spawned in parallel (one process per GPU) so all three
# 48 GB cards on caribou are saturated at the same time. Compared to
# run_local_mixed.sh's stream A/B split (which idles GPU 2 once Stream B's
# single model finishes), this script ignores the TP=2 path entirely and
# treats every (model, dataset) as TP=1 on the GPU it was given.
#
# Use it for re-runs / extension passes where every model fits on a
# single 48 GB card (true for all our retained models at FP8 / MXFP4).
# For initial runs that include gpt-oss-120b (63 GB needs TP=2),
# use run_local_mixed.sh instead.
#
# Required:
#   --gpu  N            single GPU index (e.g. 0)
#   --models a,b,c      comma list of model names to run on that GPU
#
# Optional:
#   --datasets x,y      restrict to a comma list of datasets
#   --limit N           instances per dataset (default 5)
#   --force             pass --force to evaluate_models (rebuild JSONs)
#
# Usage:
#   ./experiments/run_per_gpu.sh --gpu 0 --models deepseek-r1-distill-70b \
#       --datasets headqa,maud --force --limit 1000
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

GPU=""
MODELS=""
SELECTED_DATASETS=""
LIMIT="5"
FORCE=""

_require_arg() {
    if [[ $# -lt 2 || -z "${2:-}" || "${2:0:2}" == "--" ]]; then
        echo "[ABORT] $1 requires a value" >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)      _require_arg "$@"; GPU="$2"; shift 2 ;;
        --models)   _require_arg "$@"; MODELS="$2"; shift 2 ;;
        --datasets) _require_arg "$@"; SELECTED_DATASETS="$2"; shift 2 ;;
        --limit)    _require_arg "$@"; LIMIT="$2"; shift 2 ;;
        --force)    FORCE="--force"; shift ;;
        -h|--help)  sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$GPU" || -z "$MODELS" ]]; then
    echo "[ABORT] both --gpu N and --models a,b,c are required" >&2
    exit 1
fi

DATASETS=(financebench finqa tatqa headqa cuad maud contractnli)
[[ -f "$PROJECT_DIR/dataset/insurance_text_simplifications_annotated.jsonl" ]] && DATASETS+=(judgebert)

if [[ -n "$SELECTED_DATASETS" ]]; then
    FILTERED=()
    for d in "${DATASETS[@]}"; do
        [[ ",$SELECTED_DATASETS," == *",$d,"* ]] && FILTERED+=("$d")
    done
    if [[ ${#FILTERED[@]} -eq 0 ]]; then
        echo "[ABORT] --datasets '$SELECTED_DATASETS' did not match any of:" \
             "${DATASETS[*]}" >&2
        exit 1
    fi
    DATASETS=("${FILTERED[@]}")
fi

IFS=',' read -ra MODEL_LIST <<< "$MODELS"

LOG_DIR="$PROJECT_DIR/experiments/results/smoke_logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

echo "============================================================"
echo " run_per_gpu.sh  GPU=$GPU  TP=1"
echo "============================================================"
echo "  Models   : ${MODEL_LIST[*]}"
echo "  Datasets : ${DATASETS[*]}"
echo "  Limit    : $LIMIT"
echo "  Force    : ${FORCE:-(no)}"
echo "============================================================"

for model in "${MODEL_LIST[@]}"; do
    log="$LOG_DIR/pergpu_${TS}_gpu${GPU}_${model}.log"
    echo "  [$(date +%H:%M:%S)] $model on GPU $GPU -> $log"
    for ds in "${DATASETS[@]}"; do
        t0=$(date +%s)
        if stdbuf -oL -eL env PYTHONUNBUFFERED=1 \
                PYTHONPATH=src python3 -u -m experiments.evaluate_models \
                --dataset "$ds" --model "$model" \
                --limit "$LIMIT" --gpu "$GPU" \
                --tensor-parallel-size 1 $FORCE \
                >> "$log" 2>&1; then
            echo "  [$(date +%H:%M:%S)] $model x $ds OK ($(( $(date +%s) - t0 ))s)"
        else
            echo "  [$(date +%H:%M:%S)] $model x $ds FAIL ($(( $(date +%s) - t0 ))s) -- see $log"
        fi
    done
done

echo "[$(date +%H:%M:%S)] GPU $GPU done"
