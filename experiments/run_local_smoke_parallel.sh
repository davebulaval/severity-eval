#!/usr/bin/env bash
# =============================================================================
# run_local_smoke_parallel.sh -- Smoke the 10 local models in parallel on 3 GPUs
#
# Distributes the local models across GPUs so that the slowest path on any
# GPU is roughly equal to the others. The two 70B models are pinned to
# different GPUs (each saturates a 48 GB card). Reasoning models, which use
# 16x more generation budget, are spread out.
#
# Usage:
#   ./experiments/run_local_smoke_parallel.sh                  # default 3 GPUs
#   ./experiments/run_local_smoke_parallel.sh --gpus 0,1       # 2 GPUs only
#   ./experiments/run_local_smoke_parallel.sh --skip granite-3.2-8b
#   ./experiments/run_local_smoke_parallel.sh --limit 10       # 10 inst per dataset
#
# Each GPU produces its own log under experiments/results/smoke_logs/.
# After all GPUs finish, a consolidated per-(model, dataset) summary is
# printed.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# ---- defaults ---------------------------------------------------------------
GPUS="0,1,2"
LIMIT="5"
SKIP_MODELS=""   # comma-separated

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)   GPUS="$2"; shift 2 ;;
        --limit)  LIMIT="$2"; shift 2 ;;
        --skip)   SKIP_MODELS="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

IFS=',' read -ra GPU_LIST <<< "$GPUS"
N_GPU=${#GPU_LIST[@]}
if [[ $N_GPU -lt 1 ]]; then
    echo "[ABORT] no GPUs selected"; exit 1
fi

# ---- model assignment -------------------------------------------------------
# Each bucket should take roughly the same wall-clock time.
# Approximate per-model smoke runtime (10 datasets x 5 instances):
#   70B non-think  : ~30-40 min
#   70B think      : ~60-90 min
#   32B think (qwq): ~40 min
#   30B MoE think  : ~30 min
#   27B            : ~20 min
#   24B            : ~15 min
#   14B think      : ~30 min
#   14B            : ~10 min
#   8-9B           : ~8 min
#
# Default 3-GPU distribution targets ~75 min per GPU:
#   GPU A: llama-3.3-70b           + qwen3-14b + gemma-2-9b
#   GPU B: deepseek-r1-distill-70b + phi-4
#   GPU C: qwq-32b + qwen3-30b-a3b + mistral-small-3 + gemma-2-27b + granite-3.2-8b

# bucket index → space-separated model list
BUCKET_0=(llama-3.3-70b qwen3-14b gemma-2-9b)
BUCKET_1=(deepseek-r1-distill-70b phi-4)
BUCKET_2=(qwq-32b qwen3-30b-a3b mistral-small-3 gemma-2-27b granite-3.2-8b)

# Fallback for fewer GPUs: collapse buckets.
if [[ $N_GPU -eq 1 ]]; then
    BUCKET_0=("${BUCKET_0[@]}" "${BUCKET_1[@]}" "${BUCKET_2[@]}")
    BUCKET_1=()
    BUCKET_2=()
elif [[ $N_GPU -eq 2 ]]; then
    BUCKET_0=("${BUCKET_0[@]}" "${BUCKET_2[@]}")
    BUCKET_1=("${BUCKET_1[@]}")
    BUCKET_2=()
fi

# Apply --skip
filter_skip() {
    local skip="$1"; shift
    local out=()
    for m in "$@"; do
        if [[ ! ",$skip," =~ ,$m, ]]; then
            out+=("$m")
        fi
    done
    printf '%s\n' "${out[@]}"
}

if [[ -n "$SKIP_MODELS" ]]; then
    mapfile -t BUCKET_0 < <(filter_skip "$SKIP_MODELS" "${BUCKET_0[@]}")
    mapfile -t BUCKET_1 < <(filter_skip "$SKIP_MODELS" "${BUCKET_1[@]}")
    mapfile -t BUCKET_2 < <(filter_skip "$SKIP_MODELS" "${BUCKET_2[@]}")
fi

# ---- show plan --------------------------------------------------------------
echo "============================================================"
echo " PARALLEL LOCAL SMOKE"
echo "============================================================"
echo "  GPUs    : $GPUS  (n=$N_GPU)"
echo "  Limit   : $LIMIT instances per dataset"
echo "  Skip    : ${SKIP_MODELS:-(none)}"
echo ""
echo "  GPU ${GPU_LIST[0]}: ${BUCKET_0[*]:-(empty)}"
if [[ $N_GPU -ge 2 ]]; then
    echo "  GPU ${GPU_LIST[1]}: ${BUCKET_1[*]:-(empty)}"
fi
if [[ $N_GPU -ge 3 ]]; then
    echo "  GPU ${GPU_LIST[2]}: ${BUCKET_2[*]:-(empty)}"
fi
echo "============================================================"
echo ""

LOG_DIR="$PROJECT_DIR/experiments/results/smoke_logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

PIDS=()
launch_bucket() {
    local gpu="$1"; shift
    local models=("$@")
    if [[ ${#models[@]} -eq 0 ]]; then
        echo "[GPU $gpu] no models, skipping"
        return
    fi
    local csv
    csv=$(IFS=','; echo "${models[*]}")
    local log="$LOG_DIR/parallel_${TS}_gpu${gpu}.log"
    echo "[GPU $gpu] launching: $csv  (log: $log)"
    (
        ./experiments/run_local_smoke.sh \
            --gpu "$gpu" --limit "$LIMIT" --models "$csv" \
            > "$log" 2>&1
        echo "[GPU $gpu] DONE (exit=$?)"
    ) &
    PIDS+=($!)
}

START_TIME=$(date +%s)

launch_bucket "${GPU_LIST[0]}" "${BUCKET_0[@]}"
if [[ $N_GPU -ge 2 ]]; then
    launch_bucket "${GPU_LIST[1]}" "${BUCKET_1[@]}"
fi
if [[ $N_GPU -ge 3 ]]; then
    launch_bucket "${GPU_LIST[2]}" "${BUCKET_2[@]}"
fi

echo ""
echo "All buckets launched. Waiting for completion ..."
echo "Tail the logs in another terminal with:"
echo "  tail -f $LOG_DIR/parallel_${TS}_gpu*.log"
echo ""

# Wait for all and collect statuses.
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAILED=$((FAILED + 1))
    fi
done

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

echo ""
echo "============================================================"
echo " PARALLEL SMOKE DONE"
echo "============================================================"
echo "  Wall time : $((ELAPSED / 60))m $((ELAPSED % 60))s"
echo "  Failures  : $FAILED / ${#PIDS[@]} GPU buckets"
echo "  Logs      : $LOG_DIR/parallel_${TS}_gpu*.log"
echo "============================================================"
echo ""

# Consolidated table across every model in every bucket.
ALL_MODELS=("${BUCKET_0[@]}" "${BUCKET_1[@]}" "${BUCKET_2[@]}")
DATASETS=(financebench finqa tatqa medcalc medqa headqa cuad maud contractnli)
[[ -f "$PROJECT_DIR/dataset/all_manual_evaluations.jsonl" ]] && DATASETS+=(rag_insurance)
[[ -f "$PROJECT_DIR/dataset/insurance_text_simplifications_annotated.jsonl" ]] && DATASETS+=(judgebert)

MODELS_PY="["$(printf '"%s",' "${ALL_MODELS[@]}" | sed 's/,$//')"]"
DATASETS_PY="["$(printf '"%s",' "${DATASETS[@]}" | sed 's/,$//')"]"

echo "Consolidated summary (across all GPUs):"
PYTHONPATH=src python3 - <<PY
import json, os
from collections import Counter

models = $MODELS_PY
datasets = $DATASETS_PY

print(f"{'Model':<28} {'Dataset':<14} {'n':>4} {'acc':>7} {'methods'}")
print("-" * 95)
totals = {}
for m in models:
    n_total = 0
    n_correct = 0
    for d in datasets:
        path = f"experiments/results/{d}_{m}.json"
        if not os.path.exists(path):
            print(f"{m:<28} {d:<14} MISS")
            continue
        try:
            data = json.load(open(path))
        except Exception as e:
            print(f"{m:<28} {d:<14} READ-ERR {e}")
            continue
        n = len(data)
        nc = sum(1 for r in data if r.get("correct"))
        methods = Counter(r.get("score_method") for r in data)
        acc = (nc / n * 100) if n else 0.0
        n_total += n
        n_correct += nc
        print(f"{m:<28} {d:<14} {n:>4} {acc:>6.1f}% {dict(methods)}")
    if n_total:
        totals[m] = (n_correct, n_total, 100 * n_correct / n_total)

print()
print("Per-model average accuracy:")
for m, (nc, n, acc) in totals.items():
    print(f"  {m:<28} {nc}/{n} = {acc:.1f}%")
PY

exit $FAILED
