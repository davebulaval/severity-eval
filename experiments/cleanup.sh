#!/usr/bin/env bash
# =============================================================================
# cleanup.sh -- Stop in-flight smoke/eval runs and tidy up artifacts
#
# By default runs in --dry-run mode: prints what would be killed/deleted
# without touching anything. Pass --apply to actually do it.
#
# Sections (toggle each independently):
#   --kill          stop running smoke/eval/python processes
#   --batch-errors  delete result JSONs that are 100% batch_error
#   --logs          delete smoke logs older than 1 day
#   --wandb-local   purge local wandb/ run dirs older than 1 day
#   --hf-cache      show HF cache size (only deletes if --purge-hf passed)
#   --purge-hf      actually delete the HF model cache (DANGEROUS, frees ~150 GB)
#   --gpu-reset     try a soft GPU reset (nvidia-smi --gpu-reset; needs sudo)
#   --all           equivalent to --kill --batch-errors --logs --wandb-local
#
# Usage:
#   ./experiments/cleanup.sh                       # dry-run, all sections
#   ./experiments/cleanup.sh --apply --all         # actually clean
#   ./experiments/cleanup.sh --apply --kill        # only stop processes
#   ./experiments/cleanup.sh --apply --batch-errors
#   ./experiments/cleanup.sh --apply --purge-hf    # +145 GB freed (re-downloads needed)
# =============================================================================
set -uo pipefail

cd "$(dirname "$0")/.."

APPLY=false
DO_KILL=false
DO_BATCH=false
DO_LOGS=false
DO_WANDB=false
DO_HF=false
DO_PURGE_HF=false
DO_GPU_RESET=false
DO_ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --apply)         APPLY=true; shift ;;
        --kill)          DO_KILL=true; shift ;;
        --batch-errors)  DO_BATCH=true; shift ;;
        --logs)          DO_LOGS=true; shift ;;
        --wandb-local)   DO_WANDB=true; shift ;;
        --hf-cache)      DO_HF=true; shift ;;
        --purge-hf)      DO_HF=true; DO_PURGE_HF=true; shift ;;
        --gpu-reset)     DO_GPU_RESET=true; shift ;;
        --all)           DO_ALL=true; shift ;;
        -h|--help)
            sed -n '2,28p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ "$DO_ALL" == "true" ]]; then
    DO_KILL=true; DO_BATCH=true; DO_LOGS=true; DO_WANDB=true
fi
# If no section was selected, default to showing all of them in dry-run.
if [[ "$DO_KILL" == "false" && "$DO_BATCH" == "false" && "$DO_LOGS" == "false" \
   && "$DO_WANDB" == "false" && "$DO_HF" == "false" && "$DO_GPU_RESET" == "false" ]]; then
    DO_KILL=true; DO_BATCH=true; DO_LOGS=true; DO_WANDB=true; DO_HF=true
fi

MODE="DRY-RUN (use --apply to actually do it)"
[[ "$APPLY" == "true" ]] && MODE="APPLY"

echo "============================================================"
echo " CLEANUP — mode: $MODE"
echo "============================================================"

# -----------------------------------------------------------------------------
# 1. Stop running processes
# -----------------------------------------------------------------------------
if [[ "$DO_KILL" == "true" ]]; then
    echo ""
    echo "## Running smoke/eval/python processes"
    echo ""
    # Match the wrappers, the python evaluators AND the vLLM/wandb
    # children they spawn. Without EngineCore + wandb-core, killed runs
    # leave zombie GPU+VRAM allocations -- exactly the failure mode that
    # required manual `pkill -9 -f EngineCore` in past sessions.
    PATTERN='(run_local_sequential_tp|run_full_pipeline|run_all\.sh|evaluate_models|test_hypotheses|validate_severity_llm|EngineCore|VLLM::EngineCore|wandb-core)'
    PIDS=$(pgrep -af "$PATTERN" | grep -v "cleanup.sh" | awk '{print $1}' || true)
    if [[ -z "$PIDS" ]]; then
        echo "(no matching processes)"
    else
        echo "Will kill:"
        pgrep -af "$PATTERN" | grep -v "cleanup.sh"
        if [[ "$APPLY" == "true" ]]; then
            echo ""
            echo "Sending SIGTERM ..."
            echo "$PIDS" | xargs -r kill 2>/dev/null || true
            sleep 3
            REMAIN=$(pgrep -af "$PATTERN" | grep -v "cleanup.sh" | awk '{print $1}' || true)
            if [[ -n "$REMAIN" ]]; then
                echo "Survived SIGTERM, sending SIGKILL ..."
                echo "$REMAIN" | xargs -r kill -9 2>/dev/null || true
                sleep 1
            fi
            echo "Done. Remaining:"
            pgrep -af "$PATTERN" | grep -v "cleanup.sh" || echo "(none)"
        fi
    fi
fi

# -----------------------------------------------------------------------------
# 2. Delete result JSONs that are 100% batch_error
# -----------------------------------------------------------------------------
if [[ "$DO_BATCH" == "true" ]]; then
    echo ""
    echo "## Result JSONs with 100% batch_error"
    echo ""
    PYTHONPATH=src python3 - <<PY
import glob, json, os, sys
from collections import Counter

apply_flag = $( [[ "$APPLY" == "true" ]] && echo "True" || echo "False" )

bad = []
for f in sorted(glob.glob("experiments/results/*_*.json")):
    try:
        data = json.load(open(f))
    except Exception:
        continue
    if not isinstance(data, list) or not data:
        continue
    methods = Counter(r.get("score_method", "?") for r in data)
    if methods.get("batch_error", 0) == len(data):
        bad.append(f)
        print(f"  {f}  ({len(data)} batch_error)")

print(f"\nTotal: {len(bad)} files")
if apply_flag and bad:
    for f in bad:
        os.remove(f)
    print(f"Deleted {len(bad)} file(s).")
PY
fi

# -----------------------------------------------------------------------------
# 3. Smoke logs older than 1 day
# -----------------------------------------------------------------------------
if [[ "$DO_LOGS" == "true" ]]; then
    echo ""
    echo "## Smoke logs older than 1 day"
    echo ""
    OLD_LOGS=$(find experiments/results/smoke_logs -name "*.log" -mtime +1 2>/dev/null)
    if [[ -z "$OLD_LOGS" ]]; then
        echo "(none)"
    else
        echo "$OLD_LOGS" | head -20
        n=$(echo "$OLD_LOGS" | wc -l)
        echo "Total: $n files"
        if [[ "$APPLY" == "true" ]]; then
            echo "$OLD_LOGS" | xargs -r rm
            echo "Deleted."
        fi
    fi
fi

# -----------------------------------------------------------------------------
# 4. Local wandb/ run dirs older than 1 day
# -----------------------------------------------------------------------------
if [[ "$DO_WANDB" == "true" ]]; then
    echo ""
    echo "## Local wandb run directories older than 1 day"
    echo ""
    if [[ -d wandb ]]; then
        OLD_WB=$(find wandb -maxdepth 1 -mindepth 1 -type d -mtime +1 2>/dev/null)
        if [[ -z "$OLD_WB" ]]; then
            echo "(none)"
        else
            echo "$OLD_WB" | head -20
            n=$(echo "$OLD_WB" | wc -l)
            total_size=$(echo "$OLD_WB" | xargs -r du -sh 2>/dev/null | tail -1 | awk '{print $1}')
            echo "Total: $n dirs (~${total_size:-?})"
            if [[ "$APPLY" == "true" ]]; then
                echo "$OLD_WB" | xargs -r rm -rf
                echo "Deleted."
            fi
        fi
    else
        echo "(no wandb/ dir)"
    fi
fi

# -----------------------------------------------------------------------------
# 5. HuggingFace cache (read-only by default, --purge-hf to actually clean)
# -----------------------------------------------------------------------------
if [[ "$DO_HF" == "true" ]]; then
    echo ""
    echo "## HuggingFace model cache size"
    echo ""
    HF_HOME_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
    if [[ -d "$HF_HOME_DIR" ]]; then
        du -sh "$HF_HOME_DIR" 2>/dev/null
        echo ""
        echo "Per-model top 10:"
        find "$HF_HOME_DIR/hub" -maxdepth 1 -mindepth 1 -type d 2>/dev/null \
            | xargs -r du -sh 2>/dev/null | sort -rh | head -10
        if [[ "$DO_PURGE_HF" == "true" && "$APPLY" == "true" ]]; then
            echo ""
            echo "PURGING $HF_HOME_DIR/hub ..."
            rm -rf "$HF_HOME_DIR/hub"
            echo "Done. Next run will re-download all models."
        elif [[ "$DO_PURGE_HF" == "true" ]]; then
            echo "(dry-run, would purge $HF_HOME_DIR/hub)"
        fi
    else
        echo "(HF cache not found at $HF_HOME_DIR)"
    fi
fi

# -----------------------------------------------------------------------------
# 6. GPU soft reset
# -----------------------------------------------------------------------------
if [[ "$DO_GPU_RESET" == "true" ]]; then
    echo ""
    echo "## GPU reset (clears stranded VRAM from killed processes)"
    echo ""
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
    if [[ "$APPLY" == "true" ]]; then
        echo "Attempting nvidia-smi --gpu-reset (needs sudo, ignores if denied)..."
        for gpu in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
            sudo -n nvidia-smi --gpu-reset -i "$gpu" 2>/dev/null || \
                echo "  GPU $gpu reset skipped (no sudo or in use)"
        done
        nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
    fi
fi

echo ""
echo "============================================================"
if [[ "$APPLY" == "false" ]]; then
    echo " Dry-run complete. Re-run with --apply to actually clean."
else
    echo " Cleanup applied."
fi
echo "============================================================"
