# Local inference (vLLM)

vLLM is the only inference engine for the local-model bucket. It provides
continuous batching, PagedAttention KV cache, and prefix caching.

The checkpoints we load are the `unsloth/X-unsloth-bnb-4bit`
HF repositories -- Unsloth-published Dynamic 2.0 quants that upcast
sensitive layers (~1-2% accuracy gain). The "-unsloth-" prefix is just
the publisher namespace on Hugging Face; vLLM reads the safetensors
directly. The unsloth Python library is **not** installed and not
required.

## Install

One-shot install + validation:

```bash
./experiments/setup_env.sh                       # default venv .severity
./experiments/setup_env.sh --venv .severity_2    # custom name
```

The script is idempotent (re-running only patches what's missing). It:
1. Creates the venv if absent (requires Python >= 3.10)
2. `pip install -r requirements.txt` (torch, vllm, bnb, API clients, ...)
3. `pip install nvidia-cuda-nvcc` (CUDA compiler, needed by flashinfer JIT
   inside vLLM -- the engine crashes mid-init without it)
4. Patches `$VENV/bin/activate` with the right `LD_LIBRARY_PATH` (cu13 /
   cu12 runtime libs) and `CUDA_HOME` + `PATH` (nvcc) exports
5. Runs `experiments/check_env.sh` to validate the whole stack

After it finishes, in any new shell:

```bash
source <venv-name>/bin/activate
```

Manual validation alone (no install):

```bash
./experiments/check_env.sh
```

Requires:
- Python 3.10+ (3.11 recommended)
- PyTorch 2.4+ (currently 2.11+cu130 in the venv)
- CUDA 12.x or 13.0 (we run 13.0; do NOT upgrade to 13.2 -- Unsloth
  reports gibberish outputs on that toolkit, NVIDIA is working on it)

## Usage

### Run a single (model, dataset) pair

```bash
PYTHONPATH=src python -m experiments.evaluate_models \
    --dataset medqa --model qwen3-14b \
    --limit 100 --gpu 0
```

### Run the smoke matrix

```bash
# Run every local model sequentially with tensor parallelism across all GPUs:
./experiments/run_local_sequential_tp.sh --gpus 0,1,2 --limit 100
```

### Benchmark a single model

```bash
PYTHONPATH=src python -m experiments.bench_inference \
    --model qwen3-14b --n-samples 8 --gpu 0
```

The script writes `experiments/benchmarks/<branch>_<commit>_<model>_<ts>.json`
with mean / p50 / p90 latency, tokens/sec, peak VRAM, and a generation
config snapshot.

Compare two benchmark runs:

```bash
PYTHONPATH=src python -m experiments.bench_inference --compare \
    experiments/benchmarks/main_*_qwen3-14b_*.json \
    experiments/benchmarks/speedup_*_qwen3-14b_*.json
```

## Troubleshooting: flashinfer JIT crash at EngineCore init

If the micro-inference fails with one of these symptoms:

```
ptxas fatal : Unsupported .version 9.2; current version is '9.1'
fatal error: nv/target: No such file or directory
ninja: build stopped: subcommand failed
```

flashinfer is JIT-compiling its sampler kernels at first run and hitting
a PTX-version mismatch in the venv-only CUDA toolkit. Fix by disabling
flashinfer's sampler and using vLLM's native PyTorch sampler:

```bash
./experiments/fix_vllm_runtime.sh
```

This kills zombies, clears the flashinfer cache, exports
`VLLM_USE_FLASHINFER_SAMPLER=0` into the venv's activate, and runs a
smoke test to confirm. `setup_env.sh` calls it automatically at the end
of install, so most users won't invoke it manually.

## Troubleshooting: flashinfer `ninja: build stopped`

If `check_env.sh` section 11 fails with

```
(EngineCore) /bin/sh: 1: /usr/local/cuda/bin/nvcc: not found
(EngineCore) ninja: build stopped: subcommand failed.
```

flashinfer is hard-coding the system CUDA path inside its JIT build,
ignoring `CUDA_HOME`. Run :

```bash
./experiments/fix_flashinfer_nvcc.sh
```

The script clears the flashinfer cache, then either:
- tries `sudo ln -sf $CUDA_HOME /usr/local/cuda` (the clean fix), or
- builds a per-user shim and persists `CUDA_PATH` + `TORCH_CUDA_HOME`
  into `$VENV/bin/activate` (no sudo needed).

It re-runs `check_env.sh` automatically. `setup_env.sh` already calls
this script at the end, so most users won't need to invoke it manually.

## Known limitations

1. **Engine lifetime**: vLLM claims a fixed fraction of GPU memory at
   load. We destroy the previous engine before loading a new one.
   Loading 11 datasets x 10 models = 10 engine loads (not 110), since
   the same engine is reused across all datasets of a given model.

2. **bnb-4bit checkpoint compatibility**: some Dynamic 2.0 `-unsloth-bnb-4bit`
   checkpoints carry config fields that vLLM's bnb loader rejects with
   `ValueError`. The loader falls back to the standard `-bnb-4bit`
   variant automatically with a warning.

3. **No assisted_decoding**: vLLM has its own speculative-decoding
   stack (`--speculative-model` on the server, `SpecDecodeWorker` in
   the engine). The HF `assistant_model` flow is not wired up here; if
   you need spec decoding, switch to vLLM server mode and point it at
   Qwen3-0.6B.

4. **CUDA 13.2 warning**: Unsloth's docs note that CUDA 13.2 produces
   gibberish outputs. We run CUDA 13.0, so this does not apply to our
   setup -- but if anyone bumps the toolkit, watch for it.
