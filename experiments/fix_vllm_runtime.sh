#!/usr/bin/env bash
# =============================================================================
# fix_vllm_runtime.sh -- Resolve the vLLM EngineCore JIT-compile crash by
# disabling flashinfer's sampler (we use vLLM's native PyTorch sampler).
#
# Background:
#   On a venv-only CUDA toolkit, nvidia-cuda-nvcc ships an nvcc that emits
#   PTX 9.2 while the bundled ptxas only understands PTX 9.0/9.1 -- so
#   anything that JIT-compiles CUDA kernels at runtime crashes. vLLM's
#   flashinfer-based sampler is the worst offender because it JIT-builds on
#   first inference.
#
#   The fix:
#     1. export VLLM_USE_FLASHINFER_SAMPLER=0  -> vLLM picks forward_native
#        (PyTorch implementation, no JIT)
#     2. evaluate_local_vllm.py passes SamplingParams without top_p, so
#        even forward_cuda's "if k or p is set" branch picks the native
#        fallback (belt-and-braces against the same crash)
#
# What this script does (idempotent):
#   1. Kill any zombie vllm / EngineCore processes left over from earlier
#      crashed runs
#   2. Clear ~/.cache/flashinfer/ (stale build.ninja with bad nvcc path)
#   3. Append VLLM_USE_FLASHINFER_SAMPLER=0 to $VENV/bin/activate
#      (guarded by marker, so re-run is a no-op)
#   4. Source the venv to pick up the new export
#   5. Run the micro-inference smoke (granite-3.2-8b on medqa --limit 1)
#      with a generous 5-minute timeout to validate the fix end-to-end
#
# Flags:
#   --skip-smoke   skip step 5 (smoke validation)
#
# Usage:
#   source <venv>/bin/activate            # already activated
#   ./experiments/fix_vllm_runtime.sh
# =============================================================================
set -uo pipefail

cd "$(dirname "$0")/.."

SKIP_SMOKE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-smoke) SKIP_SMOKE=true; shift ;;
        -h|--help)    sed -n '2,33p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo " FIX vLLM runtime: disable flashinfer sampler"
echo "============================================================"

# -----------------------------------------------------------------------------
# 0. Sanity: venv active
# -----------------------------------------------------------------------------
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "[ABORT] no venv activated. Run: source <venv>/bin/activate"
    exit 1
fi
echo "  venv: $VIRTUAL_ENV"

# -----------------------------------------------------------------------------
# 1. Kill leftover zombie processes
# -----------------------------------------------------------------------------
echo
echo "## 1. Kill leftover vllm / EngineCore processes"
KILLED=0
for pattern in "EngineCore" "evaluate_models" "VLLM::EngineCore"; do
    PIDS=$(pgrep -f "$pattern" 2>/dev/null || true)
    if [[ -n "$PIDS" ]]; then
        echo "  killing $pattern: $PIDS"
        echo "$PIDS" | xargs -r kill -9 2>/dev/null || true
        KILLED=$((KILLED+$(echo "$PIDS" | wc -w)))
    fi
done
if [[ "$KILLED" -gt 0 ]]; then
    sleep 2
    echo "  killed $KILLED process(es)"
else
    echo "  (no zombies found)"
fi

# Show GPU memory after kill
nvidia-smi --query-gpu=index,memory.used --format=csv 2>/dev/null | head -5

# -----------------------------------------------------------------------------
# 2. Clear flashinfer JIT cache
# -----------------------------------------------------------------------------
echo
echo "## 2. Clear flashinfer JIT cache"
if [[ -d "$HOME/.cache/flashinfer" ]]; then
    rm -rf "$HOME/.cache/flashinfer"
    echo "  removed $HOME/.cache/flashinfer"
else
    echo "  (cache already empty)"
fi

# -----------------------------------------------------------------------------
# 3. Patch $VENV/bin/activate with VLLM_USE_FLASHINFER_SAMPLER=0
# -----------------------------------------------------------------------------
echo
echo "## 3. Persist VLLM_USE_FLASHINFER_SAMPLER=0 in activate"
ACTIVATE="$VIRTUAL_ENV/bin/activate"
MARKER="# === severity-eval fix_vllm_runtime.sh patch ==="
if grep -q "$MARKER" "$ACTIVATE"; then
    echo "  activate already exports VLLM_USE_FLASHINFER_SAMPLER -- skipping"
else
    cat >> "$ACTIVATE" << 'PATCH'

# === severity-eval fix_vllm_runtime.sh patch ===
# flashinfer's JIT sampler crashes on venv-only CUDA toolkits (PTX
# version mismatch between bundled nvcc and ptxas). Force vLLM to use
# its native PyTorch sampler instead. See experiments/fix_vllm_runtime.sh.
export VLLM_USE_FLASHINFER_SAMPLER=0
PATCH
    echo "  appended VLLM_USE_FLASHINFER_SAMPLER=0 to $ACTIVATE"
fi

# -----------------------------------------------------------------------------
# 4. Re-source so the export is live in this shell
# -----------------------------------------------------------------------------
echo
echo "## 4. Re-source venv to apply the export"
# `deactivate` is a shell function defined inside activate; it does not
# propagate to scripts launched from the parent shell. Sourcing activate
# again directly is enough -- the second source picks up our new export
# (the activate script unsets and re-sets VIRTUAL_ENV-related vars itself).
# shellcheck source=/dev/null
source "$VIRTUAL_ENV/bin/activate"
echo "  VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-(unset)}"

# -----------------------------------------------------------------------------
# 5. Smoke: micro-inference end-to-end (with timeout)
# -----------------------------------------------------------------------------
if [[ "$SKIP_SMOKE" == "true" ]]; then
    echo
    echo "## 5. Smoke micro-inference: SKIPPED (--skip-smoke)"
else
    echo
    echo "## 5. Smoke micro-inference (timeout 600s)"
    echo "  Loads granite-3.2-8b in vLLM and runs 1 prompt on medqa."
    echo "  Cold path: 1-3 min for model load + warmup, plus 60-120s of"
    echo "  wandb teardown after the prediction lands. The timeout is"
    echo "  generous so the kill does not race the cleanup."
    echo
    LOG=/tmp/fix_vllm_runtime.log
    if timeout 600 env PYTHONPATH=src python3 -m experiments.evaluate_models \
            --dataset medqa --model granite-3.2-8b \
            --limit 1 --gpu 0 --force > "$LOG" 2>&1; then
        echo "  micro-inference returned exit 0"
    else
        rc=$?
        if [[ "$rc" -eq 124 ]]; then
            echo "  TIMEOUT after 600s -- see $LOG"
            # Even if timed out, the inference may have succeeded before
            # the cleanup was killed. Surface that distinction.
            if grep -q "accuracy=" "$LOG"; then
                echo "  (but accuracy was logged -- prediction succeeded;"
                echo "   wandb / Python shutdown likely got killed mid-cleanup)"
            fi
        else
            echo "  exit $rc -- see $LOG"
        fi
    fi
    echo
    echo "  --- last 20 lines of log ---"
    tail -20 "$LOG"
fi

echo
echo "============================================================"
echo " DONE"
echo "============================================================"
echo "  If the smoke shows 'accuracy=...' (any value), the fix worked."
echo "  If it timed out at the warmup stage, the engine is still booting"
echo "    -- try increasing the limit, or run check_env.sh manually."
echo "  If a new error appears, share /tmp/fix_vllm_runtime.log."
