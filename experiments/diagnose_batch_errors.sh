#!/usr/bin/env bash
# =============================================================================
# diagnose_batch_errors.sh -- Pull the exact failure mode for batch_error runs
#
# We see lots of "score_method: batch_error" but evaluate_models.py swallows
# the underlying exception. This script extracts:
#   1. The actual Python traceback lines from the most recent smoke logs
#   2. The first error context per (gpu, model)
#   3. The batch_error count per (model, dataset) JSON
#   4. nvidia-smi snapshot to spot residual VRAM pressure
#
# Usage:
#   ./experiments/diagnose_batch_errors.sh
# =============================================================================
set -uo pipefail

cd "$(dirname "$0")/.."

LOG_DIR="experiments/results/smoke_logs"
RES_DIR="experiments/results"
OUT="/tmp/batch_error_diagnostic.md"

{
echo "# Diagnostic batch_error"
echo
echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')"
echo

# ---- (1) Smoke logs that exist --------------------------------------------
echo "## Smoke logs disponibles"
echo
echo '```'
ls -lh "$LOG_DIR"/parallel_*.log 2>/dev/null | tail -20
echo '```'
echo

# Find the LATEST parallel run
LATEST_TS=$(ls "$LOG_DIR"/parallel_*_gpu0.log 2>/dev/null | sort | tail -1 | sed -E 's|.*/parallel_([0-9_]+)_gpu0\.log|\1|')
if [[ -z "$LATEST_TS" ]]; then
    echo "**Aucun log smoke trouvé.**"
    exit 0
fi
echo "Latest run timestamp: \`$LATEST_TS\`"
echo

# ---- (2) Error messages per GPU log ---------------------------------------
echo "## Erreurs détectées (grep across all GPU logs of latest run)"
echo
for gpu_log in "$LOG_DIR"/parallel_${LATEST_TS}_gpu*.log; do
    [[ -f "$gpu_log" ]] || continue
    gpu=$(basename "$gpu_log" .log)
    echo "### $gpu"
    echo
    echo '```'
    # Capture first ERROR + context around it
    grep -nE "ERROR|OutOfMemoryError|CUDA out of memory|RuntimeError|Failed|Traceback" "$gpu_log" \
        | head -30 || echo "(no errors detected)"
    echo '```'
    echo
done

# ---- (3) Traceback excerpts (longest context) -----------------------------
echo "## Tracebacks complets (premiers 80 lignes par GPU)"
echo
for gpu_log in "$LOG_DIR"/parallel_${LATEST_TS}_gpu*.log; do
    [[ -f "$gpu_log" ]] || continue
    gpu=$(basename "$gpu_log" .log)
    # Extract from first "Traceback" or "ERROR" to ~80 lines after
    first_err=$(grep -nE "Traceback|^[0-9:]+ \[ERROR\]" "$gpu_log" | head -1 | cut -d: -f1)
    if [[ -n "$first_err" ]]; then
        echo "### $gpu (first error at line $first_err)"
        echo
        echo '```'
        sed -n "${first_err},$((first_err + 80))p" "$gpu_log"
        echo '```'
        echo
    fi
done

# ---- (4) batch_error count per (model, dataset) JSON ----------------------
echo "## batch_error count par fichier de résultats"
echo
echo "| Fichier | n | batch_error | empty | autre |"
echo "|---|---:|---:|---:|---|"
PYTHONPATH=src python3 - <<'PY'
import glob, json, os
from collections import Counter
files = sorted(glob.glob("experiments/results/*_*.json"))
rows = []
for f in files:
    name = os.path.basename(f)
    if name == "smoke_logs":
        continue
    try:
        data = json.load(open(f))
    except Exception:
        continue
    if not isinstance(data, list) or not data:
        continue
    methods = Counter(r.get("score_method", "?") for r in data)
    be = methods.get("batch_error", 0)
    em = methods.get("empty", 0)
    rest = {k: v for k, v in methods.items() if k not in ("batch_error", "empty") and v}
    # Only show files where there's a problem OR for context: limit to recent
    mtime = os.path.getmtime(f)
    rows.append((mtime, name, len(data), be, em, rest))

# sort by mtime desc, take top 40 most recent
rows.sort(reverse=True)
for _, name, n, be, em, rest in rows[:40]:
    flag = " ⚠️" if be >= n / 2 else ""
    print(f"| `{name}` | {n} | {be}{flag} | {em} | {rest} |")
PY
echo

# ---- (5) nvidia-smi snapshot ---------------------------------------------
echo "## nvidia-smi snapshot"
echo
echo '```'
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv 2>&1 || echo "(nvidia-smi failed)"
echo '```'
echo

# ---- (6) Inference config used (from logs) --------------------------------
echo "## Configs d'inférence (max_new_tokens, max_length, batch_size)"
echo
echo '```'
grep -h "Local config:" "$LOG_DIR"/parallel_${LATEST_TS}_gpu*.log 2>/dev/null \
    | sort -u | head -20 || echo "(no Local config lines found)"
echo '```'
echo

# ---- (7) Disk + memory snapshot ------------------------------------------
echo "## Disk + RAM"
echo
echo '```'
df -h /home 2>/dev/null | head -3
echo
free -h | head -3
echo '```'
echo

} > "$OUT"

echo "Diagnostic écrit dans $OUT ($(wc -l < "$OUT") lignes)"
echo
echo "Pour me coller dans le chat :"
echo "  cat $OUT"
