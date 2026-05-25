#!/usr/bin/env bash
# =============================================================================
# check_env.sh -- Validate that the local-model bucket is ready to run
#
# Runs a chain of checks before launching evaluate_models / bench_inference.
# Failures here always block the smoke / full runs, so the script exits
# non-zero on any FAIL.
#
# Checks (in order):
#   1. venv .severity activated
#   2. CUDA toolkit version
#       - warn if 13.2 (Unsloth reports gibberish outputs, NVIDIA bug)
#       - error if missing
#   3. NVIDIA driver + nvidia-smi
#   4. LD_LIBRARY_PATH points at nvidia/cu13/lib (CUDA 13 runtime libs)
#   5. Persistence patch in .severity/bin/activate
#   6. severity_eval imports cleanly
#   7. torch + CUDA visibility (cuda.is_available, GPU count, compute cap)
#   8. bitsandbytes native lib loads (no libnvJitLink error)
#   9. nvcc + CUDA_HOME (flashinfer JIT compile inside vLLM needs them)
#  10. vLLM imports (LLM, SamplingParams)
#  11. Optional micro-inference smoke (1 prompt, granite-3.2-8b)
#
# Flags:
#   --fix         append the LD_LIBRARY_PATH export to .severity/bin/activate
#                 if missing
#   --gpu N       pin the smoke micro-inference to GPU N (default 0)
#   --skip-micro  skip step 10 (saves ~60 s on cold caches)
#
# Usage:
#   ./experiments/check_env.sh
#   ./experiments/check_env.sh --fix --gpu 0
#   ./experiments/check_env.sh --skip-micro
# =============================================================================
set -uo pipefail

cd "$(dirname "$0")/.."

FIX=false
GPU="0"
SKIP_MICRO=false

_require_arg() {
    if [[ $# -lt 2 || -z "${2:-}" || "${2:0:2}" == "--" ]]; then
        echo "[ABORT] $1 requires a value" >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --fix)        FIX=true; shift ;;
        --gpu)        _require_arg "$@"; GPU="$2"; shift 2 ;;
        --skip-micro) SKIP_MICRO=true; shift ;;
        -h|--help)    sed -n '2,32p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Detect the python3.X directory inside the venv (the site-packages live
# under lib/pythonX.Y/, so hardcoding 3.11 breaks on 3.10 / 3.12 venvs).
PY_LIB_DIR=""
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PY_LIB_DIR=$(find "$VIRTUAL_ENV/lib" -maxdepth 1 -mindepth 1 \
        -type d -name "python3.*" 2>/dev/null | sort | tail -1)
fi

PASS=0
FAIL=0
WARN=0

ok()   { echo "  OK    $*"; PASS=$((PASS+1)); }
fail() { echo "  FAIL  $*"; FAIL=$((FAIL+1)); }
warn() { echo "  WARN  $*"; WARN=$((WARN+1)); }

echo "============================================================"
echo " ENV DIAGNOSTIC -- severity-eval (vLLM backend)"
echo "============================================================"

# -----------------------------------------------------------------------------
# 1. venv .severity
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
# 2. CUDA toolkit
# -----------------------------------------------------------------------------
echo
echo "## 2. CUDA toolkit"
CUDA_VERSION=""
if python3 -c "import torch; print(torch.version.cuda)" >/tmp/check_env_cuda_v.out 2>&1; then
    CUDA_VERSION=$(cat /tmp/check_env_cuda_v.out)
fi
if [[ -z "$CUDA_VERSION" || "$CUDA_VERSION" == "None" ]]; then
    if command -v nvcc >/dev/null 2>&1; then
        CUDA_VERSION=$(nvcc --version 2>/dev/null \
            | grep -oE "release [0-9]+\.[0-9]+" | awk '{print $2}')
    fi
fi
if [[ -z "$CUDA_VERSION" || "$CUDA_VERSION" == "None" ]]; then
    fail "CUDA toolkit not detected (torch.version.cuda is None and nvcc missing)"
elif [[ "$CUDA_VERSION" == "13.2"* ]]; then
    fail "CUDA toolkit is 13.2 -- known to produce gibberish outputs under"
    echo "        vLLM. NVIDIA is working on a fix."
    echo "        Downgrade to 13.0 or 13.1, or roll back to 12.x."
elif [[ "$CUDA_VERSION" == "13."* ]]; then
    ok "CUDA toolkit $CUDA_VERSION (13.0 / 13.1 are known-good)"
elif [[ "$CUDA_VERSION" == "12."* ]]; then
    ok "CUDA toolkit $CUDA_VERSION (12.x supported)"
else
    warn "CUDA toolkit $CUDA_VERSION (untested by this repo; expected 12.x or 13.0/13.1)"
fi

# -----------------------------------------------------------------------------
# 3. NVIDIA driver + nvidia-smi
# -----------------------------------------------------------------------------
echo
echo "## 3. NVIDIA driver / nvidia-smi"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi not in PATH"
else
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    N_GPU=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
    if [[ -z "$DRIVER" ]]; then
        fail "nvidia-smi present but returns no GPUs"
    else
        ok "driver $DRIVER, $N_GPU GPU(s) visible"
        nvidia-smi --query-gpu=index,name,compute_cap,memory.total \
            --format=csv,noheader 2>/dev/null | while IFS= read -r line; do
            echo "        GPU: $line"
        done
    fi
fi

# -----------------------------------------------------------------------------
# 4. LD_LIBRARY_PATH for CUDA runtime libs
# -----------------------------------------------------------------------------
echo
echo "## 4. LD_LIBRARY_PATH"
EXPECTED_CU13=""
EXPECTED_CU12=""
if [[ -n "$PY_LIB_DIR" ]]; then
    EXPECTED_CU13="$PY_LIB_DIR/site-packages/nvidia/cu13/lib"
    EXPECTED_CU12="$PY_LIB_DIR/site-packages/nvidia/cu12/lib"
fi
if [[ -z "$PY_LIB_DIR" ]]; then
    warn "skipped (no venv active, cannot anchor cuXX path against \$VIRTUAL_ENV)"
elif [[ -d "$EXPECTED_CU13" ]] \
   && [[ ":${LD_LIBRARY_PATH:-}:" == *":$EXPECTED_CU13:"* ]]; then
    ok "LD_LIBRARY_PATH contains $EXPECTED_CU13"
elif [[ -d "$EXPECTED_CU12" ]] \
   && [[ ":${LD_LIBRARY_PATH:-}:" == *":$EXPECTED_CU12:"* ]]; then
    ok "LD_LIBRARY_PATH contains $EXPECTED_CU12 (CUDA 12 toolkit)"
elif [[ -d "$EXPECTED_CU13" ]]; then
    fail "LD_LIBRARY_PATH missing $EXPECTED_CU13"
    echo "        current LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-(unset)}"
elif [[ -d "$EXPECTED_CU12" ]]; then
    fail "LD_LIBRARY_PATH missing $EXPECTED_CU12"
    echo "        current LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-(unset)}"
else
    warn "nvidia/cu13 and nvidia/cu12 lib dirs both missing -- pure-pip torch?"
fi

# -----------------------------------------------------------------------------
# 5. Persistence patch in the *currently active* venv's activate
# -----------------------------------------------------------------------------
echo
echo "## 5. Persistence in \$VIRTUAL_ENV/bin/activate"
# Target the active venv's activate, not a hard-coded path -- the user may
# have several venvs (.severity, .severity_2, etc.) and we want to patch
# the one that is actually being sourced.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    warn "skipped (no venv active, cannot locate bin/activate)"
    ACTIVATE=""
else
    ACTIVATE="$VIRTUAL_ENV/bin/activate"
fi
if [[ -n "$ACTIVATE" && ! -f "$ACTIVATE" ]]; then
    fail "$ACTIVATE missing"
elif [[ -n "$ACTIVATE" ]]; then
    if grep -q "nvidia/cu13/lib\|nvidia/cu12/lib" "$ACTIVATE"; then
        ok "activate exports LD_LIBRARY_PATH for nvidia/cuXX libs"
    else
        warn "activate does NOT export LD_LIBRARY_PATH for nvidia/cuXX"
        if [[ "$FIX" == "true" ]]; then
            echo "        appending export ..."
            # The patch uses a shell glob at activation time so it auto-discovers
            # the Python version (3.10 / 3.11 / 3.12) inside the venv.
            cat <<'PATCH' >> "$ACTIVATE"

# === added by experiments/check_env.sh --fix ===
for _cudir in "$VIRTUAL_ENV"/lib/python3.*/site-packages/nvidia/cu13/lib \
              "$VIRTUAL_ENV"/lib/python3.*/site-packages/nvidia/cu12/lib; do
    if [ -d "$_cudir" ]; then
        export LD_LIBRARY_PATH="$_cudir:${LD_LIBRARY_PATH:-}"
        break
    fi
done
unset _cudir
PATCH
            ok "appended. Re-source the venv: deactivate; source $ACTIVATE"
        else
            echo "        (re-run with --fix to append it automatically)"
        fi
    fi
fi

# -----------------------------------------------------------------------------
# 6. severity_eval import
# -----------------------------------------------------------------------------
echo
echo "## 6. severity_eval import"
if PYTHONPATH=src python3 -c "import severity_eval; print(severity_eval.__version__)" \
        >/tmp/check_env_se.out 2>&1; then
    ok "severity_eval $(cat /tmp/check_env_se.out)"
else
    fail "severity_eval import broken:"
    sed 's/^/        /' /tmp/check_env_se.out
fi

# -----------------------------------------------------------------------------
# 7. torch + CUDA
# -----------------------------------------------------------------------------
echo
echo "## 7. torch + CUDA visibility"
if python3 -c "
import torch
assert torch.cuda.is_available(), 'cuda.is_available() returned False'
n = torch.cuda.device_count()
caps = [torch.cuda.get_device_capability(i) for i in range(n)]
caps_str = ', '.join(f'{a}.{b}' for a, b in caps)
print(f'{n} GPU(s), torch {torch.__version__}, cuda {torch.version.cuda}, compute_cap=[{caps_str}]')
" >/tmp/check_env_cuda.out 2>&1; then
    ok "$(cat /tmp/check_env_cuda.out)"
else
    fail "torch.cuda diagnostic failed:"
    sed 's/^/        /' /tmp/check_env_cuda.out
fi

# -----------------------------------------------------------------------------
# 8. bitsandbytes
# -----------------------------------------------------------------------------
echo
echo "## 8. bitsandbytes native lib"
if python3 -c "import bitsandbytes; print(bitsandbytes.__version__)" \
        >/tmp/check_env_bnb.out 2>&1; then
    out=$(cat /tmp/check_env_bnb.out)
    if grep -qi "library load error\|libnvJitLink\|libcublas" /tmp/check_env_bnb.out; then
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
# 9. nvcc + CUDA_HOME (flashinfer JIT compile)
# -----------------------------------------------------------------------------
echo
echo "## 9. nvcc + CUDA_HOME (flashinfer JIT)"
# vLLM uses flashinfer for its sampling kernels; flashinfer JIT-compiles
# them at first run via nvcc. Without nvcc on PATH (and CUDA_HOME pointing
# at the toolkit) the engine crashes mid-init with
#   /bin/sh: 1: /usr/local/cuda/bin/nvcc: not found
if command -v nvcc >/dev/null 2>&1; then
    nvcc_v=$(nvcc --version 2>/dev/null | grep -oE "release [0-9]+\.[0-9]+" | awk '{print $2}')
    if [[ "$nvcc_v" == "13.2"* ]]; then
        fail "nvcc release $nvcc_v -- 13.2 is known to fail flashinfer JIT"
        echo "        (gibberish outputs + ninja build crashes; NVIDIA WIP)"
        echo "        Downgrade: pip install 'nvidia-cuda-nvcc<13.2' --upgrade --force-reinstall"
    else
        ok "nvcc on PATH: $(command -v nvcc) (release $nvcc_v)"
    fi
else
    fail "nvcc not on PATH -- flashinfer JIT will crash at engine init"
    echo "        Run ./experiments/setup_env.sh, or:"
    echo "          pip install 'nvidia-cuda-nvcc<13.2'"
    echo "          export CUDA_HOME=\$VIRTUAL_ENV/lib/python3.*/site-packages/nvidia/cuda_nvcc"
    echo "          export PATH=\$CUDA_HOME/bin:\$PATH"
fi
if [[ -n "${CUDA_HOME:-}" && -d "$CUDA_HOME" ]]; then
    ok "CUDA_HOME set: $CUDA_HOME"
elif command -v nvcc >/dev/null 2>&1; then
    warn "CUDA_HOME unset (some libs still need it explicitly)"
fi

# -----------------------------------------------------------------------------
# 10. vLLM
# -----------------------------------------------------------------------------
echo
echo "## 10. vLLM import"
if python3 -c "from vllm import LLM, SamplingParams; import vllm; print(vllm.__version__)" \
        >/tmp/check_env_vllm.out 2>&1; then
    ok "vllm $(cat /tmp/check_env_vllm.out)"
else
    fail "vllm import broken:"
    sed 's/^/        /' /tmp/check_env_vllm.out
fi

# -----------------------------------------------------------------------------
# 11. Micro-inference smoke (optional)
# -----------------------------------------------------------------------------
if [[ "$SKIP_MICRO" == "true" ]]; then
    echo
    echo "## 11. End-to-end micro-inference: SKIPPED (--skip-micro)"
elif [[ $FAIL -gt 0 ]]; then
    echo
    echo "## 11. End-to-end micro-inference: SKIPPED (earlier FAIL detected)"
else
    echo
    echo "## 11. End-to-end micro-inference (granite-3.2-8b on medqa --limit 1)"
    echo "        (this exercises the full vLLM load path; can take 1-2 min cold)"
    echo
    if PYTHONPATH=src python3 -m experiments.evaluate_models \
            --dataset medqa --model granite-3.2-8b \
            --limit 1 --gpu "$GPU" --force >/tmp/check_env_micro.out 2>&1; then
        if grep -q "accuracy=" /tmp/check_env_micro.out; then
            ok "micro-inference ran:"
            grep -E "accuracy=|batch_error|ERROR" /tmp/check_env_micro.out \
                | head -3 | sed 's/^/        /'
            if grep -q "batch_error" /tmp/check_env_micro.out; then
                warn "micro produced batch_error -- model loaded but inference failed"
            fi
        else
            warn "micro completed but no accuracy line found:"
            tail -10 /tmp/check_env_micro.out | sed 's/^/        /'
        fi
    else
        fail "micro-inference crashed:"
        tail -20 /tmp/check_env_micro.out | sed 's/^/        /'
    fi
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo
echo "============================================================"
echo "  PASS=$PASS  FAIL=$FAIL  WARN=$WARN"
if [[ $FAIL -gt 0 ]]; then
    echo "  Status: NOT READY -- fix the FAIL items before running smoke"
    if [[ "$FIX" != "true" ]]; then
        echo "  Tip: try re-running with --fix (auto-appends LD_LIBRARY_PATH)"
    fi
    exit 1
elif [[ $WARN -gt 0 ]]; then
    echo "  Status: READY (with warnings)"
else
    echo "  Status: READY -- smoke / full runs should work"
fi
echo "============================================================"
