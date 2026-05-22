#!/usr/bin/env bash
# =============================================================================
# check_env.sh -- Diagnose why the smoke buckets exit in 0 seconds
#
# Runs a chain of checks that mirror what run_local_smoke.sh needs:
#   1. venv is active
#   2. LD_LIBRARY_PATH points at the nvidia/cu13 libs
#   3. The persistence patch is in .severity/bin/activate
#   4. severity_eval imports cleanly
#   5. bitsandbytes native library loads
#   6. unsloth imports
#   7. transformers imports
#   8. A 1-instance evaluate_models.py call actually produces output
#
# Pass --fix to add the LD_LIBRARY_PATH export to .severity/bin/activate
# if it's missing.
#
# Pass --gpu N to pin the test inference to GPU N (default 0).
#
# Usage:
#   ./experiments/check_env.sh
#   ./experiments/check_env.sh --fix
#   ./experiments/check_env.sh --fix --gpu 1
# =============================================================================
set -uo pipefail

cd "$(dirname "$0")/.."

FIX=false
GPU="0"
while [[ $# -gt 0 ]]; do
    case $1 in
        --fix) FIX=true; shift ;;
        --gpu) GPU="$2"; shift 2 ;;
        -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

PASS=0
FAIL=0
WARN=0

ok()   { echo "  OK    $*"; PASS=$((PASS+1)); }
fail() { echo "  FAIL  $*"; FAIL=$((FAIL+1)); }
warn() { echo "  WARN  $*"; WARN=$((WARN+1)); }

echo "============================================================"
echo " ENV DIAGNOSTIC"
echo "============================================================"

# -----------------------------------------------------------------------------
# 1. venv
# -----------------------------------------------------------------------------
echo
echo "## 1. venv .severity"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    fail "no venv activated. Run: source .severity/bin/activate"
elif [[ "${VIRTUAL_ENV}" != *".severity"* ]]; then
    warn "VIRTUAL_ENV is $VIRTUAL_ENV (expected .severity). Other venv?"
else
    ok "venv active: $VIRTUAL_ENV"
fi

# -----------------------------------------------------------------------------
# 2. LD_LIBRARY_PATH (current shell)
# -----------------------------------------------------------------------------
echo
echo "## 2. LD_LIBRARY_PATH for CUDA 13"
EXPECTED_LIB="$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/cu13/lib"
if [[ -d "$EXPECTED_LIB" ]]; then
    if [[ ":${LD_LIBRARY_PATH:-}:" == *":$EXPECTED_LIB:"* ]]; then
        ok "LD_LIBRARY_PATH contains $EXPECTED_LIB"
    else
        fail "LD_LIBRARY_PATH missing $EXPECTED_LIB"
        echo "        current LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-(unset)}"
    fi
else
    fail "expected dir does not exist: $EXPECTED_LIB"
fi

# -----------------------------------------------------------------------------
# 3. Persistence patch in .severity/bin/activate
# -----------------------------------------------------------------------------
echo
echo "## 3. Persistence in .severity/bin/activate"
ACTIVATE=".severity/bin/activate"
if [[ ! -f "$ACTIVATE" ]]; then
    fail "$ACTIVATE missing"
else
    if grep -q "nvidia/cu13/lib" "$ACTIVATE"; then
        ok "activate already exports LD_LIBRARY_PATH for cu13"
    else
        warn "activate does NOT export LD_LIBRARY_PATH for cu13"
        if [[ "$FIX" == "true" ]]; then
            echo "        appending export ..."
            cat <<'PATCH' >> "$ACTIVATE"

# === added by experiments/check_env.sh --fix ===
if [ -d "$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/cu13/lib" ]; then
    export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
fi
PATCH
            ok "appended. Re-source the venv: deactivate; source .severity/bin/activate"
        else
            echo "        (re-run with --fix to append it automatically)"
        fi
    fi
fi

# -----------------------------------------------------------------------------
# 4. severity_eval import
# -----------------------------------------------------------------------------
echo
echo "## 4. severity_eval import"
if PYTHONPATH=src python3 -c "import severity_eval; print(severity_eval.__version__)" >/tmp/check_env_se.out 2>&1; then
    ok "severity_eval $(cat /tmp/check_env_se.out)"
else
    fail "severity_eval import broken:"
    sed 's/^/        /' /tmp/check_env_se.out
fi

# -----------------------------------------------------------------------------
# 5. bitsandbytes native lib
# -----------------------------------------------------------------------------
echo
echo "## 5. bitsandbytes"
if python3 -c "import bitsandbytes; print(bitsandbytes.__version__)" >/tmp/check_env_bnb.out 2>&1; then
    out=$(cat /tmp/check_env_bnb.out)
    if grep -q "library load error\|libnvJitLink" /tmp/check_env_bnb.out; then
        fail "bitsandbytes failed to load native lib:"
        sed 's/^/        /' /tmp/check_env_bnb.out
    else
        ok "bitsandbytes $out"
    fi
else
    fail "bitsandbytes import broken:"
    sed 's/^/        /' /tmp/check_env_bnb.out
fi

# -----------------------------------------------------------------------------
# 6. unsloth + transformers
# -----------------------------------------------------------------------------
echo
echo "## 6. unsloth + transformers"
if python3 -c "import unsloth; import transformers; print(f'unsloth {unsloth.__version__}, transformers {transformers.__version__}')" >/tmp/check_env_us.out 2>&1; then
    ok "$(cat /tmp/check_env_us.out)"
else
    warn "unsloth/transformers import had output:"
    sed 's/^/        /' /tmp/check_env_us.out
fi

# -----------------------------------------------------------------------------
# 7. CUDA visibility
# -----------------------------------------------------------------------------
echo
echo "## 7. CUDA / GPU"
if python3 -c "import torch; assert torch.cuda.is_available(); print(f'{torch.cuda.device_count()} GPU(s), torch {torch.__version__}, cuda {torch.version.cuda}')" >/tmp/check_env_cuda.out 2>&1; then
    ok "$(cat /tmp/check_env_cuda.out)"
else
    fail "torch.cuda not available:"
    sed 's/^/        /' /tmp/check_env_cuda.out
fi

# -----------------------------------------------------------------------------
# 8. End-to-end micro-inference (1 instance, tiny model)
# -----------------------------------------------------------------------------
echo
echo "## 8. End-to-end micro-inference (granite-3.2-8b on medqa --limit 1)"
echo "        (this exercises the same path as the smoke; <2 min)"
echo
if PYTHONPATH=src python3 -m experiments.evaluate_models \
        --dataset medqa --model granite-3.2-8b \
        --limit 1 --gpu "$GPU" --force >/tmp/check_env_micro.out 2>&1; then
    if grep -q "accuracy=" /tmp/check_env_micro.out; then
        ok "micro-inference ran:"
        grep -E "accuracy=|batch_error|ERROR" /tmp/check_env_micro.out | head -3 | sed 's/^/        /'
        if grep -q "batch_error" /tmp/check_env_micro.out; then
            warn "micro produced batch_error — model loaded but inference crashed"
        fi
    else
        warn "micro completed but no accuracy line found:"
        tail -10 /tmp/check_env_micro.out | sed 's/^/        /'
    fi
else
    fail "micro-inference crashed:"
    tail -20 /tmp/check_env_micro.out | sed 's/^/        /'
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo
echo "============================================================"
echo "  PASS=$PASS  FAIL=$FAIL  WARN=$WARN"
if [[ $FAIL -gt 0 ]]; then
    echo "  Status: NOT READY — fix the FAIL items before running smoke"
    if [[ "$FIX" != "true" ]]; then
        echo "  Tip: try re-running with --fix"
    fi
    exit 1
elif [[ $WARN -gt 0 ]]; then
    echo "  Status: READY (with warnings)"
else
    echo "  Status: READY — smoke should run"
fi
echo "============================================================"
