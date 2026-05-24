#!/usr/bin/env bash
# =============================================================================
# run_local_smoke.sh -- Smoke test: local models on a tiny slice (5 inst/dataset)
#
# Goal: prove every local model loads, infers, scores and writes a JSON before
# committing to the multi-day full run. Estimated total wall time on a single
# 48 GB GPU: 30-60 minutes for all 10 local models on all 11 datasets.
#
# Strategy:
#   - Iterate the 10 local models in ASCENDING size order (smallest first).
#   - Cap each dataset at 5 instances via --limit 5.
#   - Use the smallest model (granite-3.2-8b, ~5 GB) first to validate the
#     end-to-end pipeline before loading 70 GB worth of weights.
#   - Each model is evicted before the next loads (already handled by
#     _load_local_model).
#   - Results land in experiments/results/<dataset>_<model>.json.
#   - Failures don't abort: the loop logs and continues.
#
# Usage:
#   ./experiments/run_local_smoke.sh                  # default GPU 0
#   ./experiments/run_local_smoke.sh --gpu 1
#   ./experiments/run_local_smoke.sh --limit 10       # broader smoke test
#   ./experiments/run_local_smoke.sh --skip-70b       # skip the two 70B models
#   ./experiments/run_local_smoke.sh --models phi-4,gemma-2-9b
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Ascending memory order: 8B -> 9B -> 14B -> 14B -> 24B -> 27B -> 30B MoE -> 32B -> 70B -> 70B
DEFAULT_MODELS=(
    granite-3.2-8b
    gemma-2-9b
    phi-4
    qwen3-14b
    mistral-small-3
    gemma-2-27b
    qwen3-30b-a3b
    qwq-32b
    llama-3.3-70b
    deepseek-r1-distill-70b
)

GPU="0"
LIMIT="5"
SKIP_70B=false
SELECTED_MODELS=""
BACKEND="hf"

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)       GPU="$2"; shift 2 ;;
        --limit)     LIMIT="$2"; shift 2 ;;
        --skip-70b)  SKIP_70B=true; shift ;;
        --models)    SELECTED_MODELS="$2"; shift 2 ;;
        --backend)   BACKEND="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Build model list
if [[ -n "$SELECTED_MODELS" ]]; then
    IFS=',' read -ra MODELS <<< "$SELECTED_MODELS"
else
    MODELS=("${DEFAULT_MODELS[@]}")
    if [[ "$SKIP_70B" == "true" ]]; then
        MODELS=("${MODELS[@]/llama-3.3-70b}")
        MODELS=("${MODELS[@]/deepseek-r1-distill-70b}")
        # Remove empty strings
        FILTERED=()
        for m in "${MODELS[@]}"; do [[ -n "$m" ]] && FILTERED+=("$m"); done
        MODELS=("${FILTERED[@]}")
    fi
fi

# Datasets to smoke (skip private ones unless their files exist)
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
if [[ -f "$PROJECT_DIR/dataset/all_manual_evaluations.jsonl" ]]; then
    DATASETS+=(rag_insurance)
fi
if [[ -f "$PROJECT_DIR/dataset/insurance_text_simplifications_annotated.jsonl" ]]; then
    DATASETS+=(judgebert)
fi

echo "============================================================"
echo " LOCAL SMOKE TEST"
echo "============================================================"
echo "  GPU      : $GPU"
echo "  Limit    : $LIMIT instances per dataset"
echo "  Models   : ${#MODELS[@]} (${MODELS[*]})"
echo "  Datasets : ${#DATASETS[@]} (${DATASETS[*]})"
echo "  Total    : $((${#MODELS[@]} * ${#DATASETS[@]})) runs"
echo "  Output   : experiments/results/<dataset>_<model>.json"
echo "============================================================"
echo ""

# Sanity: package importable, taxonomy reachable
PYTHONPATH=src python3 - <<'PY' || { echo "[ABORT] severity_eval import broken"; exit 1; }
import severity_eval
from severity_eval.taxonomy import list_domains
assert list_domains(), "no built-in taxonomies"
print(f"[OK] severity_eval v{severity_eval.__version__} loaded, taxonomies={list_domains()}")
PY

LOG_DIR="$PROJECT_DIR/experiments/results/smoke_logs"
mkdir -p "$LOG_DIR"
SMOKE_LOG="$LOG_DIR/smoke_$(date +%Y%m%d_%H%M%S).log"
echo "Smoke log: $SMOKE_LOG"
echo ""

CURRENT=0
FAILED=0
PASSED=0
TOTAL=$((${#MODELS[@]} * ${#DATASETS[@]}))
START_TIME=$(date +%s)

for model in "${MODELS[@]}"; do
    for ds in "${DATASETS[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo ""
        echo "[$CURRENT/$TOTAL] $model x $ds (limit=$LIMIT)"
        T0=$(date +%s)

        if PYTHONPATH=src python3 -m experiments.evaluate_models \
            --dataset "$ds" --model "$model" \
            --limit "$LIMIT" --gpu "$GPU" --backend "$BACKEND" --force \
            2>&1 | tee -a "$SMOKE_LOG"; then
            T1=$(date +%s)
            echo "[$CURRENT/$TOTAL] OK in $((T1 - T0))s"
            PASSED=$((PASSED + 1))
        else
            T1=$(date +%s)
            echo "[$CURRENT/$TOTAL] FAIL in $((T1 - T0))s"
            FAILED=$((FAILED + 1))
        fi

        # Free GPU memory between models (the next iteration's same-model runs
        # are fine; cross-model evictions are handled by the loader).
    done
done

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

echo ""
echo "============================================================"
echo " SMOKE TEST DONE"
echo "============================================================"
echo "  Total    : $TOTAL"
echo "  Passed   : $PASSED"
echo "  Failed   : $FAILED"
echo "  Time     : $((ELAPSED / 60))m $((ELAPSED % 60))s"
echo "  Log      : $SMOKE_LOG"
echo "============================================================"

# Quick sanity: every result file produced by this smoke run has the expected
# columns (model, prediction, correct, score_method, severity, domain).
echo ""
echo "Per-(model, dataset) summary:"

MODELS_PY="["$(printf '"%s",' "${MODELS[@]}" | sed 's/,$//')"]"
DATASETS_PY="["$(printf '"%s",' "${DATASETS[@]}" | sed 's/,$//')"]"

PYTHONPATH=src python3 - <<PY
import json, os
from collections import Counter

models = $MODELS_PY
datasets = $DATASETS_PY

print(f"{'Model':<28} {'Dataset':<14} {'n':>4} {'acc':>6} {'methods'}")
print("-" * 90)
for m in models:
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
        n_correct = sum(1 for r in data if r.get("correct"))
        methods = Counter(r.get("score_method") for r in data)
        acc = (n_correct / n * 100) if n else 0
        print(f"{m:<28} {d:<14} {n:>4} {acc:>5.1f}% {dict(methods)}")
PY

if [[ $FAILED -gt 0 ]]; then
    echo ""
    echo "WARNING: $FAILED run(s) failed. Check $SMOKE_LOG before launching the full matrix."
    exit 1
fi

echo ""
echo "All smoke runs passed. Safe to launch the full local matrix with:"
echo "  ./experiments/run_all.sh --skip-api --gpu $GPU --limit 2500"
