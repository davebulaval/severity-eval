#!/usr/bin/env bash
# =============================================================================
# run_all.sh — Run the full severity-eval experiment matrix
#
# Usage:
#   ./experiments/run_all.sh                 # All datasets × all models
#   ./experiments/run_all.sh --wandb         # With wandb logging
#   ./experiments/run_all.sh --limit 100     # Cap instances per dataset
#   ./experiments/run_all.sh --dry-run       # Show what would run
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# --- Configuration -----------------------------------------------------------

MODELS=(gpt-4o claude-sonnet llama-3-70b mistral-large gemini-pro)

# Tier 1: public datasets that auto-download (server-ready)
DATASETS_PUBLIC=(
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

# Tier 2: private / local datasets (require manual file placement)
DATASETS_PRIVATE=(
    rag_insurance
    judgebert
)

DELAY=1.0
LIMIT=""
WANDB_FLAG=""
DRY_RUN=false
SKIP_PRIVATE=false
FORCE_FLAG=""

# --- Parse arguments ---------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case $1 in
        --wandb)       WANDB_FLAG="--wandb"; shift ;;
        --limit)       LIMIT="$2"; shift 2 ;;
        --delay)       DELAY="$2"; shift 2 ;;
        --dry-run)     DRY_RUN=true; shift ;;
        --skip-private) SKIP_PRIVATE=true; shift ;;
        --force)       FORCE_FLAG="--force"; shift ;;
        -h|--help)
            echo "Usage: $0 [--wandb] [--limit N] [--delay S] [--dry-run] [--skip-private] [--force]"
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# --- Load environment --------------------------------------------------------

if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.env"
    set +a
    echo "[OK] Loaded .env"
else
    echo "[WARN] No .env file found at $PROJECT_DIR/.env"
fi

# --- Validate API keys -------------------------------------------------------

MISSING_KEYS=0
for var in OPENAI_API_KEY ANTHROPIC_API_KEY OPENROUTER_API_KEY MISTRAL_API_KEY GEMINI_API_KEY; do
    if [[ -z "${!var:-}" ]]; then
        echo "[ERROR] Missing $var"
        MISSING_KEYS=1
    else
        echo "[OK] $var is set"
    fi
done

if [[ $MISSING_KEYS -eq 1 ]]; then
    echo ""
    echo "Set missing keys in $PROJECT_DIR/.env and retry."
    exit 1
fi

# --- Check private datasets --------------------------------------------------

DATASETS_TO_RUN=("${DATASETS_PUBLIC[@]}")

if [[ "$SKIP_PRIVATE" == "false" ]]; then
    DATASET_DIR="$PROJECT_DIR/dataset"
    for ds in "${DATASETS_PRIVATE[@]}"; do
        case $ds in
            rag_insurance)
                if [[ -f "$DATASET_DIR/all_manual_evaluations.jsonl" ]]; then
                    DATASETS_TO_RUN+=("$ds")
                    echo "[OK] Private dataset '$ds' found"
                else
                    echo "[SKIP] Private dataset '$ds' — place all_manual_evaluations.jsonl in dataset/"
                fi
                ;;
            judgebert)
                if [[ -f "$DATASET_DIR/insurance_text_simplifications_annotated.jsonl" ]]; then
                    DATASETS_TO_RUN+=("$ds")
                    echo "[OK] Private dataset '$ds' found"
                else
                    echo "[SKIP] Private dataset '$ds' — place insurance_text_simplifications_annotated.jsonl in dataset/"
                fi
                ;;
        esac
    done
else
    echo "[INFO] Skipping private datasets (--skip-private)"
fi

# --- Summary -----------------------------------------------------------------

echo ""
echo "=========================================="
echo "  severity-eval — experiment matrix"
echo "=========================================="
echo "  Datasets : ${#DATASETS_TO_RUN[@]} (${DATASETS_TO_RUN[*]})"
echo "  Models   : ${#MODELS[@]} (${MODELS[*]})"
echo "  Total    : $((${#DATASETS_TO_RUN[@]} * ${#MODELS[@]})) runs"
echo "  Delay    : ${DELAY}s between API calls"
echo "  Limit    : ${LIMIT:-none (full dataset)}"
echo "  wandb    : ${WANDB_FLAG:-off}"
echo "  Force    : ${FORCE_FLAG:-off (skip existing)}"
echo "=========================================="
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN] Would execute:"
    for ds in "${DATASETS_TO_RUN[@]}"; do
        for model in "${MODELS[@]}"; do
            echo "  python -m experiments.evaluate_models --dataset $ds --model $model --delay $DELAY ${LIMIT:+--limit $LIMIT} $WANDB_FLAG $FORCE_FLAG"
        done
    done
    exit 0
fi

# --- Run experiments ---------------------------------------------------------

RESULTS_DIR="$PROJECT_DIR/experiments/results"
mkdir -p "$RESULTS_DIR"

TOTAL=$((${#DATASETS_TO_RUN[@]} * ${#MODELS[@]}))
CURRENT=0
FAILED=0
SKIPPED=0
START_TIME=$(date +%s)

LOG_FILE="$RESULTS_DIR/run_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG_FILE"

for ds in "${DATASETS_TO_RUN[@]}"; do
    for model in "${MODELS[@]}"; do
        CURRENT=$((CURRENT + 1))
        RESULT_FILE="$RESULTS_DIR/${ds}_${model}.json"

        # Skip if already done (unless --force)
        if [[ -f "$RESULT_FILE" && -z "$FORCE_FLAG" ]]; then
            echo "[$CURRENT/$TOTAL] SKIP $ds × $model (results exist)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        echo ""
        echo "[$CURRENT/$TOTAL] START $ds × $model"
        echo "  $(date '+%Y-%m-%d %H:%M:%S')"

        CMD=(python -m experiments.evaluate_models
             --dataset "$ds"
             --model "$model"
             --delay "$DELAY")

        [[ -n "$LIMIT" ]] && CMD+=(--limit "$LIMIT")
        [[ -n "$WANDB_FLAG" ]] && CMD+=("$WANDB_FLAG")
        [[ -n "$FORCE_FLAG" ]] && CMD+=("$FORCE_FLAG")

        if "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"; then
            echo "[$CURRENT/$TOTAL] DONE  $ds × $model"
        else
            echo "[$CURRENT/$TOTAL] FAIL  $ds × $model (exit code $?)"
            FAILED=$((FAILED + 1))
            # Continue with next run — don't abort the full matrix
        fi
    done
done

# --- Summary -----------------------------------------------------------------

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS_REM=$(( ELAPSED % 60 ))

echo ""
echo "=========================================="
echo "  Experiments complete"
echo "=========================================="
echo "  Total    : $TOTAL"
echo "  Done     : $((TOTAL - FAILED - SKIPPED))"
echo "  Skipped  : $SKIPPED"
echo "  Failed   : $FAILED"
echo "  Time     : ${HOURS}h ${MINUTES}m ${SECONDS_REM}s"
echo "  Results  : $RESULTS_DIR/"
echo "  Log      : $LOG_FILE"
echo "=========================================="

# --- List result files -------------------------------------------------------

echo ""
echo "Result files:"
ls -lh "$RESULTS_DIR"/*.json 2>/dev/null || echo "  (none)"

exit $FAILED
