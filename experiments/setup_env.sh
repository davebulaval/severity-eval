#!/usr/bin/env bash
# =============================================================================
# setup_env.sh -- One-shot install + validation for the severity-eval venv
#
# Idempotent: re-running is safe. It only adds what is missing and skips
# whatever already exists.
#
# Steps:
#   1. Verify Python >= 3.10
#   2. Create the venv if it does not exist
#   3. Upgrade pip
#   4. pip install -r requirements.txt
#   5. pip install nvidia-cuda-nvcc (flashinfer JIT compile needs nvcc;
#      vLLM crashes at engine init without it)
#   6. Detect nvcc + cuXX paths inside the venv
#   7. Patch $VENV/bin/activate with LD_LIBRARY_PATH + CUDA_HOME + PATH
#      exports (guarded by a marker so a second run is a no-op)
#   8. Re-source the venv in a sub-shell and run check_env.sh
#
# Flags:
#   --venv NAME      venv directory name (default: .severity)
#   --python BIN     python binary used to create the venv (default: python3)
#   --reinstall      force `pip install --upgrade --force-reinstall` for deps
#   --skip-check     skip the final check_env.sh run
#
# Usage:
#   ./experiments/setup_env.sh                       # default .severity
#   ./experiments/setup_env.sh --venv .severity_2    # custom name
#   ./experiments/setup_env.sh --python python3.11   # explicit Python
#   ./experiments/setup_env.sh --reinstall           # rebuild from scratch
#
# After this script finishes, activate the venv in your own shell:
#   source <VENV>/bin/activate
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

VENV=".severity"
PYTHON="python3"
REINSTALL=false
SKIP_CHECK=false

_require_arg() {
    if [[ $# -lt 2 || -z "${2:-}" || "${2:0:2}" == "--" ]]; then
        echo "[ABORT] $1 requires a value" >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --venv)        _require_arg "$@"; VENV="$2"; shift 2 ;;
        --python)      _require_arg "$@"; PYTHON="$2"; shift 2 ;;
        --reinstall)   REINSTALL=true; shift ;;
        --skip-check)  SKIP_CHECK=true; shift ;;
        -h|--help)     sed -n '2,36p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo " SETUP severity-eval venv: $VENV"
echo "============================================================"

# -----------------------------------------------------------------------------
# 1. Python version
# -----------------------------------------------------------------------------
echo
echo "## 1. Python version"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "[ABORT] $PYTHON not in PATH"
    exit 1
fi
PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [[ "$PY_MAJOR" -lt 3 || ( "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10 ) ]]; then
    echo "[ABORT] Python 3.10+ required, got $PY_VERSION via $PYTHON"
    exit 1
fi
echo "  Python $PY_VERSION via $(command -v "$PYTHON")"

# -----------------------------------------------------------------------------
# 2. Create venv
# -----------------------------------------------------------------------------
echo
echo "## 2. Create venv $VENV"
if [[ -d "$VENV" ]]; then
    echo "  $VENV already exists -- reusing"
else
    "$PYTHON" -m venv "$VENV"
    echo "  Created $VENV"
fi

# -----------------------------------------------------------------------------
# 3. Activate + upgrade pip
# -----------------------------------------------------------------------------
echo
echo "## 3. Activate + upgrade pip"
# shellcheck source=/dev/null
source "$VENV/bin/activate"
pip install --upgrade pip -q
echo "  $(pip --version)"

# -----------------------------------------------------------------------------
# 4. Install requirements
# -----------------------------------------------------------------------------
echo
echo "## 4. Install requirements.txt"
if [[ "$REINSTALL" == "true" ]]; then
    echo "  --reinstall: forcing upgrade + force-reinstall"
    pip install --upgrade --force-reinstall -r requirements.txt
else
    pip install -r requirements.txt
fi
echo "  done"

# -----------------------------------------------------------------------------
# 5. Install nvidia-cuda-nvcc (flashinfer JIT dep)
# -----------------------------------------------------------------------------
echo
echo "## 5. Install nvidia-cuda-nvcc (CUDA compiler for flashinfer / vLLM JIT)"
# Pin below 13.2: the 13.2 toolkit is known to produce gibberish outputs
# and silent kernel build failures under unsloth/vLLM (Unsloth-reported
# bug, NVIDIA WIP). We want a 13.0 / 13.1 nvcc to match the runtime libs
# that torch ships with cu130.
pip install "nvidia-cuda-nvcc<13.2" -q

# -----------------------------------------------------------------------------
# 6. Detect nvcc + cuXX paths
# -----------------------------------------------------------------------------
echo
echo "## 6. Detect nvcc + CUDA runtime libs in the venv"
NVCC_PATH=$(find "$VIRTUAL_ENV" -name "nvcc" -type f 2>/dev/null | head -1)
if [[ -z "$NVCC_PATH" ]]; then
    echo "[ABORT] nvcc binary not found after pip install nvidia-cuda-nvcc"
    exit 1
fi
CUDA_HOME_DIR="$(dirname "$(dirname "$NVCC_PATH")")"
echo "  nvcc      : $NVCC_PATH"
echo "  CUDA_HOME : $CUDA_HOME_DIR"

PY_LIB_DIR=$(find "$VIRTUAL_ENV/lib" -maxdepth 1 -mindepth 1 \
    -type d -name "python3.*" 2>/dev/null | sort | tail -1)
if [[ -z "$PY_LIB_DIR" ]]; then
    echo "[ABORT] no python3.X lib dir found in $VIRTUAL_ENV/lib"
    exit 1
fi
CU_LIB_DIR=""
for _candidate in "$PY_LIB_DIR/site-packages/nvidia/cu13/lib" \
                  "$PY_LIB_DIR/site-packages/nvidia/cu12/lib"; do
    if [[ -d "$_candidate" ]]; then
        CU_LIB_DIR="$_candidate"
        break
    fi
done
if [[ -z "$CU_LIB_DIR" ]]; then
    echo "[WARN] nvidia/cu13 + nvidia/cu12 lib dirs both missing -- LD_LIBRARY_PATH"
    echo "       will not be patched. bitsandbytes may fail to load libnvJitLink."
else
    echo "  cuXX lib  : $CU_LIB_DIR"
fi

# -----------------------------------------------------------------------------
# 7. Patch activate
# -----------------------------------------------------------------------------
echo
echo "## 7. Patch $VIRTUAL_ENV/bin/activate"
ACTIVATE="$VIRTUAL_ENV/bin/activate"
PATCH_MARKER="# === severity-eval setup_env.sh patch ==="
if grep -q "$PATCH_MARKER" "$ACTIVATE"; then
    echo "  activate already patched -- skipping (idempotent)"
else
    cat >> "$ACTIVATE" << 'PATCH'

# === severity-eval setup_env.sh patch ===
# CUDA runtime libs (cu13 preferred, cu12 fallback) so bitsandbytes,
# vLLM and torch find libnvJitLink, libcublas, etc. at load time.
for _cudir in "$VIRTUAL_ENV"/lib/python3.*/site-packages/nvidia/cu13/lib \
              "$VIRTUAL_ENV"/lib/python3.*/site-packages/nvidia/cu12/lib; do
    if [ -d "$_cudir" ]; then
        export LD_LIBRARY_PATH="$_cudir:${LD_LIBRARY_PATH:-}"
        break
    fi
done
unset _cudir

# CUDA compiler (nvcc) for flashinfer / vLLM JIT.
for _nvcc in "$VIRTUAL_ENV"/lib/python3.*/site-packages/nvidia/cuda_nvcc/bin/nvcc \
            "$VIRTUAL_ENV"/lib/python3.*/site-packages/nvidia/cu13/bin/nvcc; do
    if [ -x "$_nvcc" ]; then
        CUDA_HOME="$(dirname "$(dirname "$_nvcc")")"
        export CUDA_HOME
        export PATH="$CUDA_HOME/bin:$PATH"
        break
    fi
done
unset _nvcc
PATCH
    echo "  patched"
fi

# -----------------------------------------------------------------------------
# 8. Re-source + flashinfer fix + check_env.sh
# -----------------------------------------------------------------------------
if [[ "$SKIP_CHECK" == "true" ]]; then
    echo
    echo "## 8. flashinfer fix + check_env.sh: SKIPPED (--skip-check)"
else
    echo
    echo "## 8. Re-source venv + apply flashinfer fix + run check_env.sh"
    # We have to re-source so the patched exports take effect in this shell.
    deactivate
    # shellcheck source=/dev/null
    source "$VENV/bin/activate"
    echo "  LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-(unset)}"
    echo "  CUDA_HOME=${CUDA_HOME:-(unset)}"
    echo
    # fix_flashinfer_nvcc.sh handles the /usr/local/cuda symlink (or shim)
    # and runs check_env.sh itself, so we skip a separate check call here.
    ./experiments/fix_flashinfer_nvcc.sh
fi

echo
echo "============================================================"
echo " SETUP COMPLETE"
echo "============================================================"
echo "  To activate this venv in another shell:"
echo "    source $VENV/bin/activate"
echo
echo "  To re-run the validation only:"
echo "    source $VENV/bin/activate && ./experiments/check_env.sh"
