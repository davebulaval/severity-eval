#!/usr/bin/env bash
# =============================================================================
# fix_flashinfer_nvcc.sh -- Resolve the flashinfer "/usr/local/cuda/bin/nvcc
# not found" crash on systems where the CUDA toolkit is installed inside the
# venv (via nvidia-cuda-nvcc) rather than at the system path.
#
# Background:
#   vLLM uses flashinfer for its sampling kernels. flashinfer JIT-compiles
#   those kernels at first run via ninja. Its build.ninja hard-codes
#   `/usr/local/cuda/bin/nvcc` instead of respecting CUDA_HOME, so even with
#   nvcc available in the venv (under nvidia/cu13/bin/) the ninja build
#   crashes with `code=127 ... /bin/sh: 1: /usr/local/cuda/bin/nvcc: not found`.
#
# What this script does (idempotent):
#   1. Verify a venv is active and CUDA_HOME is set
#   2. Clear the flashinfer cache (~/.cache/flashinfer/) so the rebuilt
#      .ninja files use the new toolchain
#   3. Try sudo symlink `/usr/local/cuda -> $CUDA_HOME` (the clean fix)
#   4. If sudo is unavailable, build a per-user shim at ~/.cuda-shim/ and
#      export CUDA_PATH / TORCH_CUDA_HOME pointing at it. Persists those
#      exports into $VIRTUAL_ENV/bin/activate (guarded by a marker).
#   5. Run check_env.sh to validate that section 11 now passes.
#
# Flags:
#   --skip-check   skip the final check_env.sh run
#
# Usage:
#   source .severity/bin/activate         # any venv with nvcc installed
#   ./experiments/fix_flashinfer_nvcc.sh
# =============================================================================
set -uo pipefail

cd "$(dirname "$0")/.."

SKIP_CHECK=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-check) SKIP_CHECK=true; shift ;;
        -h|--help)    sed -n '2,28p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo " FIX flashinfer/vLLM nvcc path crash"
echo "============================================================"

# -----------------------------------------------------------------------------
# 1. Prereqs: venv + CUDA_HOME
# -----------------------------------------------------------------------------
echo
echo "## 1. Prereqs"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "[ABORT] no venv activated. Run: source <venv>/bin/activate"
    exit 1
fi
echo "  venv: $VIRTUAL_ENV"

if [[ -z "${CUDA_HOME:-}" || ! -d "${CUDA_HOME:-}" ]]; then
    # Try to auto-detect from the venv
    CUDA_HOME=$(find "$VIRTUAL_ENV" -maxdepth 6 -type d \
        \( -path "*/nvidia/cu13" -o -path "*/nvidia/cu12" -o -path "*/nvidia/cuda_nvcc" \) \
        2>/dev/null | head -1)
    if [[ -z "$CUDA_HOME" || ! -x "$CUDA_HOME/bin/nvcc" ]]; then
        echo "[ABORT] CUDA_HOME not set and could not auto-detect a venv nvcc."
        echo "        Run ./experiments/setup_env.sh first."
        exit 1
    fi
    export CUDA_HOME
fi
if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    echo "[ABORT] $CUDA_HOME/bin/nvcc not executable (or missing)."
    exit 1
fi
echo "  CUDA_HOME: $CUDA_HOME"
echo "  nvcc:      $CUDA_HOME/bin/nvcc"

# -----------------------------------------------------------------------------
# 2. Clear flashinfer JIT cache
# -----------------------------------------------------------------------------
echo
echo "## 2. Clear flashinfer cache"
FLASH_CACHE="$HOME/.cache/flashinfer"
if [[ -d "$FLASH_CACHE" ]]; then
    rm -rf "$FLASH_CACHE"
    echo "  removed $FLASH_CACHE"
else
    echo "  (cache already empty)"
fi

# -----------------------------------------------------------------------------
# 3. Try sudo symlink /usr/local/cuda -> $CUDA_HOME
# -----------------------------------------------------------------------------
echo
echo "## 3. Symlink /usr/local/cuda -> $CUDA_HOME"
NEED_SHIM=true
if [[ -L "/usr/local/cuda" ]]; then
    EXISTING=$(readlink -f /usr/local/cuda)
    if [[ "$EXISTING" == "$(readlink -f "$CUDA_HOME")" ]]; then
        echo "  symlink already points at the venv CUDA -- nothing to do"
        NEED_SHIM=false
    else
        echo "  /usr/local/cuda exists and points at $EXISTING (different)"
    fi
elif [[ -e "/usr/local/cuda" ]]; then
    echo "  /usr/local/cuda exists and is NOT a symlink -- leaving it alone"
elif sudo -n true 2>/dev/null; then
    echo "  passwordless sudo available, creating symlink ..."
    if sudo ln -sf "$CUDA_HOME" /usr/local/cuda; then
        echo "  created /usr/local/cuda -> $CUDA_HOME"
        NEED_SHIM=false
    else
        echo "  sudo ln failed -- falling back to per-user shim"
    fi
else
    echo "  no passwordless sudo -- falling back to per-user shim"
fi

# -----------------------------------------------------------------------------
# 4. Per-user shim if /usr/local/cuda is not usable
# -----------------------------------------------------------------------------
echo
echo "## 4. Per-user shim (if needed)"
if [[ "$NEED_SHIM" == "true" ]]; then
    SHIM="$HOME/.cuda-shim"
    mkdir -p "$SHIM/bin" "$SHIM/include" "$SHIM/lib"
    ln -sfn "$CUDA_HOME/bin/nvcc" "$SHIM/bin/nvcc"
    # Mirror include + lib via directory-level symlinks where possible
    if [[ -d "$CUDA_HOME/include" ]]; then
        ln -sfn "$CUDA_HOME/include" "$SHIM/include.real" 2>/dev/null || true
    fi
    if [[ -d "$CUDA_HOME/lib" ]]; then
        ln -sfn "$CUDA_HOME/lib" "$SHIM/lib.real" 2>/dev/null || true
    fi
    echo "  shim at $SHIM (bin/nvcc symlinked)"
    export CUDA_PATH="$CUDA_HOME"
    export TORCH_CUDA_HOME="$CUDA_HOME"
    echo "  exported CUDA_PATH=$CUDA_PATH"
    echo "  exported TORCH_CUDA_HOME=$TORCH_CUDA_HOME"

    # Persist into activate (guarded by marker, idempotent)
    ACTIVATE="$VIRTUAL_ENV/bin/activate"
    MARKER="# === severity-eval fix_flashinfer_nvcc.sh patch ==="
    if grep -q "$MARKER" "$ACTIVATE"; then
        echo "  activate already patched -- skipping"
    else
        cat >> "$ACTIVATE" << 'PATCH'

# === severity-eval fix_flashinfer_nvcc.sh patch ===
# flashinfer hard-codes /usr/local/cuda/bin/nvcc in its ninja build.
# When the toolkit lives in the venv, /usr/local/cuda is missing.
# Export both CUDA_PATH and TORCH_CUDA_HOME so tools that consult those
# (instead of CUDA_HOME) also find the right path.
if [ -n "${CUDA_HOME:-}" ] && [ -d "$CUDA_HOME" ]; then
    export CUDA_PATH="$CUDA_HOME"
    export TORCH_CUDA_HOME="$CUDA_HOME"
fi
PATCH
        echo "  appended CUDA_PATH / TORCH_CUDA_HOME exports to $ACTIVATE"
    fi
else
    echo "  (skipped, system symlink is in place)"
fi

# -----------------------------------------------------------------------------
# 5. Validate
# -----------------------------------------------------------------------------
echo
if [[ "$SKIP_CHECK" == "true" ]]; then
    echo "## 5. check_env.sh: SKIPPED (--skip-check)"
else
    echo "## 5. Run check_env.sh"
    ./experiments/check_env.sh
fi

echo
echo "============================================================"
echo " FIX COMPLETE"
echo "============================================================"
if [[ "$NEED_SHIM" == "true" ]]; then
    echo "  No sudo was available. CUDA_PATH + TORCH_CUDA_HOME are now"
    echo "  persisted in $VIRTUAL_ENV/bin/activate."
    echo "  If section 11 still FAILs, ask an admin to run:"
    echo "      sudo ln -sf $CUDA_HOME /usr/local/cuda"
else
    echo "  System symlink in place; future runs will skip step 3-4."
fi
