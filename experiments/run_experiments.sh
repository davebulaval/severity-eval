#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh — Run both prompt configurations for the EMNLP paper
#
# Configuration 1: "original" — prompts from published papers (primary results)
# Configuration 2: "standard" — uniform HELM-style template (ablation study)
#
# Usage:
#   ./experiments/run_experiments.sh                    # Full matrix (both configs)
#   ./experiments/run_experiments.sh --config original  # Primary results only
#   ./experiments/run_experiments.sh --config standard  # Ablation only
#   ./experiments/run_experiments.sh --limit 100        # Cap instances per dataset
#   ./experiments/run_experiments.sh --dry-run           # Show what would run
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# --- Configuration -----------------------------------------------------------

CONFIG="both"
LIMIT=""
DELAY=2.0
DRY_RUN=false
SKIP_PRIVATE=false
SKIP_LOCAL=false
SKIP_API=false
FORCE_FLAG=""
WANDB_FLAG=""

# --- Parse arguments ---------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)       CONFIG="$2"; shift 2 ;;
        --limit)        LIMIT="$2"; shift 2 ;;
        --delay)        DELAY="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=true; shift ;;
        --skip-private) SKIP_PRIVATE=true; shift ;;
        --skip-local)   SKIP_LOCAL=true; shift ;;
        --skip-api)     SKIP_API=true; shift ;;
        --force)        FORCE_FLAG="--force"; shift ;;
        --wandb)        WANDB_FLAG="--wandb"; shift ;;
        -h|--help)
            echo "Usage: $0 [--config original|standard|both] [--limit N] [--delay S] [--dry-run] [--skip-private] [--skip-local] [--skip-api] [--force] [--wandb]"
            echo ""
            echo "  --config original   Run primary results (paper prompts) only"
            echo "  --config standard   Run ablation (HELM-style) only"
            echo "  --config both       Run both configurations (default)"
            echo "  --skip-local        Skip local vLLM models"
            echo "  --skip-api          Skip API models (run local only)"
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# --- Validate config ---------------------------------------------------------

if [[ "$CONFIG" != "original" && "$CONFIG" != "standard" && "$CONFIG" != "both" ]]; then
    echo "[ERROR] --config must be 'original', 'standard', or 'both'. Got: $CONFIG"
    exit 1
fi

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

# --- Build prompt style list -------------------------------------------------

PROMPT_STYLES=()
if [[ "$CONFIG" == "both" ]]; then
    PROMPT_STYLES=("original" "standard")
elif [[ "$CONFIG" == "original" ]]; then
    PROMPT_STYLES=("original")
else
    PROMPT_STYLES=("standard")
fi

# --- Delegate to run_all.sh --------------------------------------------------

TOTAL_FAILED=0

for style in "${PROMPT_STYLES[@]}"; do
    echo ""
    echo "######################################################################"
    echo "  CONFIGURATION: ${style}"
    echo "######################################################################"
    echo ""

    CMD=(bash "$SCRIPT_DIR/run_all.sh"
         --prompt-style "$style"
         --delay "$DELAY")

    [[ -n "$LIMIT" ]]       && CMD+=(--limit "$LIMIT")
    [[ -n "$FORCE_FLAG" ]]  && CMD+=("$FORCE_FLAG")
    [[ -n "$WANDB_FLAG" ]]  && CMD+=("$WANDB_FLAG")
    [[ "$SKIP_PRIVATE" == "true" ]] && CMD+=(--skip-private)
    [[ "$SKIP_LOCAL" == "true" ]]   && CMD+=(--skip-local)
    [[ "$SKIP_API" == "true" ]]     && CMD+=(--skip-api)

    if [[ "$DRY_RUN" == "true" ]]; then
        CMD+=(--dry-run)
    fi

    echo "[CMD] ${CMD[*]}"
    echo ""

    if "${CMD[@]}"; then
        echo ""
        echo "[OK] Configuration '${style}' completed successfully"
    else
        EXIT_CODE=$?
        echo ""
        echo "[WARN] Configuration '${style}' completed with $EXIT_CODE failure(s)"
        TOTAL_FAILED=$((TOTAL_FAILED + EXIT_CODE))
    fi
done

# --- Final summary -----------------------------------------------------------

echo ""
echo "######################################################################"
echo "  ALL EXPERIMENTS COMPLETE"
echo "######################################################################"
echo "  Configurations : ${PROMPT_STYLES[*]}"
echo "  Total failures : $TOTAL_FAILED"
echo "  Results dir    : $PROJECT_DIR/experiments/results/"
echo ""
echo "  Primary results (original):"
echo "    experiments/results/{dataset}_{model}.json"
echo ""
echo "  Ablation results (standard):"
echo "    experiments/results/{dataset}_{model}_standard.json"
echo "######################################################################"

exit $TOTAL_FAILED
