#!/usr/bin/env bash
# =============================================================================
# run_full_pipeline.sh -- End-to-end severity-eval pipeline
#
# Stages (each is idempotent and can be re-run):
#   1. Evaluate API models (batch APIs, 50% cost reduction where available).
#   2. Evaluate local open-weight models (4-bit NF4 via vLLM + bitsandbytes;
#      checkpoints are the Unsloth Dynamic 2.0 -unsloth-bnb-4bit variants).
#   3. Validate severity labels against an LLM judge (one run per dataset).
#   4. Compute actuarial metrics and run RQ1-RQ5 analyses.
#   5. Generate paper figures (PDF) and LaTeX tables.
#
# This script is intended for the dedicated compute server. Override stages
# with --skip-* to resume after a crash without redoing finished work.
#
# Usage:
#   ./experiments/run_full_pipeline.sh                      # full pipeline
#   ./experiments/run_full_pipeline.sh --limit 200          # 200 instances/dataset
#   ./experiments/run_full_pipeline.sh --skip-local         # API + analysis only
#   ./experiments/run_full_pipeline.sh --skip-eval          # analysis only (results already present)
#   ./experiments/run_full_pipeline.sh --dry-run            # print plan, do nothing
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# ---- defaults ---------------------------------------------------------------
LIMIT=""
SKIP_API=false
SKIP_LOCAL=false
SKIP_VALIDATE=true        # severity-label validation is opt-in (expensive)
SKIP_ANALYSIS=false
SKIP_FIGURES=false
SKIP_EVAL=false
DRY_RUN=false
GPU=""
BATCH_SIZE=""
FORCE_FLAG=""
PROMPT_STYLE="original"
RESULTS_DIR="$PROJECT_DIR/experiments/results"
OUTPUT_DIR="$PROJECT_DIR/results"
FIGURES_DIR="$PROJECT_DIR/paper/figures"

# ---- argument parsing -------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --limit)         LIMIT="$2"; shift 2 ;;
        --gpu)           GPU="$2"; shift 2 ;;
        --batch-size)    BATCH_SIZE="$2"; shift 2 ;;
        --prompt-style)  PROMPT_STYLE="$2"; shift 2 ;;
        --skip-api)      SKIP_API=true; shift ;;
        --skip-local)    SKIP_LOCAL=true; shift ;;
        --skip-eval)     SKIP_EVAL=true; shift ;;
        --skip-analysis) SKIP_ANALYSIS=true; shift ;;
        --skip-figures)  SKIP_FIGURES=true; shift ;;
        --with-validate) SKIP_VALIDATE=false; shift ;;
        --dry-run)       DRY_RUN=true; shift ;;
        --force)         FORCE_FLAG="--force"; shift ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---- environment ------------------------------------------------------------
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

echo "============================================================"
echo " severity-eval -- full pipeline"
echo "============================================================"
echo "  Project   : $PROJECT_DIR"
echo "  Limit     : ${LIMIT:-none}"
echo "  Prompt    : $PROMPT_STYLE"
echo "  Skip API  : $SKIP_API"
echo "  Skip local: $SKIP_LOCAL"
echo "  Skip eval : $SKIP_EVAL"
echo "  Skip val. : $SKIP_VALIDATE"
echo "  Skip anal.: $SKIP_ANALYSIS"
echo "  Skip figs.: $SKIP_FIGURES"
echo "  Force     : ${FORCE_FLAG:-no}"
echo "  GPU       : ${GPU:-auto}"
echo "============================================================"

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN] Stopping before executing stages."
    exit 0
fi

# ---- stage 1+2: evaluate models --------------------------------------------
if [[ "$SKIP_EVAL" == "false" ]]; then
    echo ""
    echo "[STAGE 1+2] Evaluating models ..."
    args=("./experiments/run_all.sh" "--prompt-style" "$PROMPT_STYLE")
    [[ -n "$LIMIT" ]]      && args+=("--limit" "$LIMIT")
    [[ -n "$GPU" ]]        && args+=("--gpu" "$GPU")
    [[ -n "$BATCH_SIZE" ]] && args+=("--batch-size" "$BATCH_SIZE")
    [[ -n "$FORCE_FLAG" ]] && args+=("$FORCE_FLAG")
    [[ "$SKIP_API" == "true" ]]   && args+=("--skip-api")
    [[ "$SKIP_LOCAL" == "true" ]] && args+=("--skip-local")
    "${args[@]}"
else
    echo "[STAGE 1+2] Skipped (--skip-eval)."
fi

# ---- stage 3: LLM-based label validation -----------------------------------
if [[ "$SKIP_VALIDATE" == "false" ]]; then
    echo ""
    echo "[STAGE 3] Validating severity labels with LLM judge ..."
    for ds in financebench finqa tatqa medcalc medqa headqa cuad maud contractnli; do
        out="$RESULTS_DIR/validation/${ds}_llm_validation.json"
        if [[ -f "$out" && -z "$FORCE_FLAG" ]]; then
            echo "  [SKIP] $ds validation already exists"
            continue
        fi
        python -m experiments.validate_severity_llm \
            --dataset "$ds" \
            ${LIMIT:+--limit "$LIMIT"} \
            || echo "  [FAIL] $ds (continuing)"
    done
else
    echo "[STAGE 3] Skipped (default; pass --with-validate to enable)."
fi

# ---- stage 4: analysis ------------------------------------------------------
if [[ "$SKIP_ANALYSIS" == "false" ]]; then
    echo ""
    echo "[STAGE 4] Computing actuarial metrics and RQ1-RQ5 ..."
    mkdir -p "$OUTPUT_DIR"
    python experiments/analysis.py \
        --results-dir "$RESULTS_DIR" \
        --output "$OUTPUT_DIR" \
        --prompt-style "$PROMPT_STYLE"
    python experiments/test_hypotheses.py \
        --results-dir "$OUTPUT_DIR" \
        || echo "  [WARN] test_hypotheses.py failed (continuing)"
else
    echo "[STAGE 4] Skipped (--skip-analysis)."
fi

# ---- stage 5: figures + tables ---------------------------------------------
if [[ "$SKIP_FIGURES" == "false" ]]; then
    echo ""
    echo "[STAGE 5] Generating figures ..."
    mkdir -p "$FIGURES_DIR"
    python experiments/figures.py \
        --results-dir "$OUTPUT_DIR" \
        --output "$FIGURES_DIR"
else
    echo "[STAGE 5] Skipped (--skip-figures)."
fi

echo ""
echo "============================================================"
echo " Done. See:"
echo "   - $OUTPUT_DIR/metrics.csv"
echo "   - $OUTPUT_DIR/hypothesis_tests.json"
echo "   - $OUTPUT_DIR/table_main.tex, table_h1.tex, table_h4.tex"
echo "   - $FIGURES_DIR/*.pdf"
echo "============================================================"
