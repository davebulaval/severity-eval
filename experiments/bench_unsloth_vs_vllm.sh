#!/usr/bin/env bash
# =============================================================================
# bench_unsloth_vs_vllm.sh -- Head-to-head benchmark between the legacy
# unsloth + HF pipeline path and the new vLLM backend on the same model
# and prompt set.
#
# Two isolated venvs are required because unsloth and vLLM have conflicting
# torch / transformers / cuda dep pins. We create them on demand.
#
# Flow:
#   1. Activate (or create) the unsloth venv; pip install torch + unsloth +
#      transformers + bitsandbytes + minimal helpers
#   2. Run experiments/bench_inference_unsloth.py --model <M> in that venv,
#      producing experiments/benchmarks/<branch>_<commit>_unsloth_<M>_<ts>.json
#   3. Activate the vllm venv (default .severity_2, or --vllm-venv arg)
#   4. Run experiments/bench_inference.py --model <M> in that venv,
#      producing experiments/benchmarks/<branch>_<commit>_<M>_<ts>.json
#   5. Print side-by-side comparison
#
# Flags:
#   --model NAME           model to benchmark (default: qwen3-14b)
#   --n-samples N          prompts per run (default: 8)
#   --max-new-tokens N     generation cap (default: 256; auto-scaled x16 for
#                          thinking models inside the bench scripts)
#   --gpu N                CUDA device (default: 0)
#   --unsloth-venv DIR     venv for the unsloth path (default: .severity)
#   --vllm-venv DIR        venv for the vLLM path    (default: .severity_2)
#   --skip-unsloth         only run the vLLM side (e.g. if unsloth is broken)
#   --skip-vllm            only run the unsloth side
#
# Usage:
#   ./experiments/bench_unsloth_vs_vllm.sh --model qwen3-14b
#   ./experiments/bench_unsloth_vs_vllm.sh --model qwq-32b --n-samples 4
# =============================================================================
set -uo pipefail

cd "$(dirname "$0")/.."

MODEL="qwen3-14b"
N_SAMPLES=8
MAX_NEW_TOKENS=256
GPU="0"
UNSLOTH_VENV=".severity"
VLLM_VENV=".severity_2"
SKIP_UNSLOTH=false
SKIP_VLLM=false

_require_arg() {
    if [[ $# -lt 2 || -z "${2:-}" || "${2:0:2}" == "--" ]]; then
        echo "[ABORT] $1 requires a value" >&2; exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)           _require_arg "$@"; MODEL="$2"; shift 2 ;;
        --n-samples)       _require_arg "$@"; N_SAMPLES="$2"; shift 2 ;;
        --max-new-tokens)  _require_arg "$@"; MAX_NEW_TOKENS="$2"; shift 2 ;;
        --gpu)             _require_arg "$@"; GPU="$2"; shift 2 ;;
        --unsloth-venv)    _require_arg "$@"; UNSLOTH_VENV="$2"; shift 2 ;;
        --vllm-venv)       _require_arg "$@"; VLLM_VENV="$2"; shift 2 ;;
        --skip-unsloth)    SKIP_UNSLOTH=true; shift ;;
        --skip-vllm)       SKIP_VLLM=true; shift ;;
        -h|--help)         sed -n '2,32p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo " UNSLOTH vs vLLM benchmark"
echo "============================================================"
echo "  model         : $MODEL"
echo "  n_samples     : $N_SAMPLES"
echo "  max_new_tokens: $MAX_NEW_TOKENS"
echo "  gpu           : $GPU"
echo "  unsloth venv  : $UNSLOTH_VENV"
echo "  vllm venv     : $VLLM_VENV"

# -----------------------------------------------------------------------------
# Helper: in a fresh subshell, source a venv and run a command
# -----------------------------------------------------------------------------
run_in_venv() {
    local venv="$1"; shift
    (
        if [[ -z "${VIRTUAL_ENV:-}" ]] || [[ "$VIRTUAL_ENV" != *"$venv"* ]]; then
            # shellcheck source=/dev/null
            source "$venv/bin/activate"
        fi
        "$@"
    )
}

UNSLOTH_JSON=""
VLLM_JSON=""

# -----------------------------------------------------------------------------
# 1. UNSLOTH side
# -----------------------------------------------------------------------------
if [[ "$SKIP_UNSLOTH" == "true" ]]; then
    echo
    echo "## UNSLOTH: SKIPPED (--skip-unsloth)"
else
    echo
    echo "## 1. UNSLOTH side ($UNSLOTH_VENV)"
    if [[ ! -d "$UNSLOTH_VENV" ]]; then
        echo "[ABORT] unsloth venv '$UNSLOTH_VENV' does not exist."
        echo "        Either pass --unsloth-venv <existing-dir>, or create it"
        echo "        beforehand with the unsloth stack:"
        echo "          python3 -m venv $UNSLOTH_VENV"
        echo "          source $UNSLOTH_VENV/bin/activate"
        echo "          pip install torch transformers accelerate bitsandbytes unsloth"
        exit 1
    fi
    # Sanity: make sure unsloth is actually importable inside the venv.
    set +e
    run_in_venv "$UNSLOTH_VENV" python3 -c "import unsloth" 2>/dev/null
    has_unsloth=$?
    set -e
    if [[ "$has_unsloth" -ne 0 ]]; then
        echo "[ABORT] unsloth is not importable inside $UNSLOTH_VENV."
        echo "        Install it: source $UNSLOTH_VENV/bin/activate && pip install unsloth"
        exit 1
    fi
    echo "  using existing venv $UNSLOTH_VENV (unsloth ok)"

    LOG=/tmp/bench_unsloth.log
    echo "  running bench_inference_unsloth.py (log: $LOG)"
    set +e
    run_in_venv "$UNSLOTH_VENV" \
        env PYTHONPATH=src python3 -m experiments.bench_inference_unsloth \
        --model "$MODEL" --n-samples "$N_SAMPLES" \
        --max-new-tokens "$MAX_NEW_TOKENS" --gpu "$GPU" \
        >"$LOG" 2>&1
    rc=$?
    set -e
    if [[ "$rc" -ne 0 ]]; then
        echo "  UNSLOTH bench failed (exit $rc) -- tail:"
        tail -20 "$LOG" | sed 's/^/    /'
    else
        # Find the freshly-written JSON
        UNSLOTH_JSON=$(ls -1t experiments/benchmarks/*unsloth_${MODEL}_*.json 2>/dev/null | head -1)
        echo "  UNSLOTH JSON: $UNSLOTH_JSON"
    fi
fi

# -----------------------------------------------------------------------------
# 2. vLLM side
# -----------------------------------------------------------------------------
if [[ "$SKIP_VLLM" == "true" ]]; then
    echo
    echo "## vLLM: SKIPPED (--skip-vllm)"
else
    echo
    echo "## 2. vLLM side ($VLLM_VENV)"
    if [[ ! -d "$VLLM_VENV" ]]; then
        echo "[ABORT] $VLLM_VENV does not exist. Run ./experiments/setup_env.sh first."
        exit 1
    fi
    set +e
    run_in_venv "$VLLM_VENV" python3 -c "import vllm" 2>/dev/null
    has_vllm=$?
    set -e
    if [[ "$has_vllm" -ne 0 ]]; then
        echo "[ABORT] vllm is not importable inside $VLLM_VENV."
        echo "        Re-run ./experiments/setup_env.sh --venv $VLLM_VENV"
        exit 1
    fi
    echo "  using existing venv $VLLM_VENV (vllm ok)"

    LOG=/tmp/bench_vllm.log
    echo "  running bench_inference.py (log: $LOG)"
    set +e
    run_in_venv "$VLLM_VENV" \
        env PYTHONPATH=src python3 -m experiments.bench_inference \
        --model "$MODEL" --n-samples "$N_SAMPLES" \
        --max-new-tokens "$MAX_NEW_TOKENS" --gpu "$GPU" \
        >"$LOG" 2>&1
    rc=$?
    set -e
    if [[ "$rc" -ne 0 ]]; then
        echo "  vLLM bench failed (exit $rc) -- tail:"
        tail -20 "$LOG" | sed 's/^/    /'
    else
        # vLLM bench writes <branch>_<commit>_<model>_<ts>.json without
        # the "unsloth" tag. Pick the newest non-unsloth match.
        VLLM_JSON=$(ls -1t experiments/benchmarks/*_${MODEL}_*.json 2>/dev/null \
            | grep -v "_unsloth_" | head -1)
        echo "  vLLM JSON   : $VLLM_JSON"
    fi
fi

# -----------------------------------------------------------------------------
# 3. Compare
# -----------------------------------------------------------------------------
echo
echo "============================================================"
echo " COMPARISON"
echo "============================================================"
if [[ -z "$UNSLOTH_JSON" || -z "$VLLM_JSON" ]]; then
    echo "  Missing one of the two JSONs -- cannot compare."
    [[ -z "$UNSLOTH_JSON" ]] && echo "    UNSLOTH_JSON not set"
    [[ -z "$VLLM_JSON"    ]] && echo "    VLLM_JSON not set"
    exit 1
fi

# Use the vllm venv's Python to parse + print the diff (it has pandas
# but we keep this stdlib-only for portability).
python3 - "$UNSLOTH_JSON" "$VLLM_JSON" <<'PY'
import json, sys
unsloth_path, vllm_path = sys.argv[1], sys.argv[2]
u = json.load(open(unsloth_path))
v = json.load(open(vllm_path))

keys = [
    ("load_seconds",         "lower"),
    ("total_seconds",        "lower"),
    ("mean_latency_seconds", "lower"),
    ("p50_latency_seconds",  "lower"),
    ("p90_latency_seconds",  "lower"),
    ("tokens_per_second",    "higher"),
    ("peak_vram_gb",         "lower"),
]
print(f"  model       : {u['model_name']}")
print(f"  n_samples   : {u['n_samples']}")
print(f"  unsloth.json: {unsloth_path}")
print(f"  vllm.json   : {vllm_path}")
print()
print(f"{'metric':<24} {'unsloth':>14} {'vllm':>14} {'delta':>14}  speedup")
print("-" * 80)
for key, want in keys:
    uv = u.get(key, float('nan'))
    vv = v.get(key, float('nan'))
    if uv and uv != 0:
        delta_pct = (vv - uv) / uv * 100
        if want == "lower":
            speedup = uv / vv if vv else float('inf')
        else:
            speedup = vv / uv if uv else float('inf')
    else:
        delta_pct = float('nan')
        speedup = float('nan')
    print(f"{key:<24} {uv:>14.2f} {vv:>14.2f} {delta_pct:>+12.1f}%  {speedup:>7.2f}x")
PY
