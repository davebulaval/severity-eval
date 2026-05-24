# vLLM backend

vLLM is an alternative inference engine for the local-model bucket. It
provides continuous batching, PagedAttention, and prefix caching, which
together are 5-30x faster than the unsloth + HF pipeline path used by
default.

This document covers the install, the gotchas, and how to switch back
to HF if vLLM does not load a checkpoint.

## When to use vLLM vs HF

| Use HF (`--backend hf`)                | Use vLLM (`--backend vllm`)            |
|----------------------------------------|----------------------------------------|
| You have not installed vLLM yet        | Long thinking-model runs               |
| You need use_cache=False (debugging)   | Many prompts per (model, dataset) pair |
| The checkpoint fails to load in vLLM   | Production runs at `--limit 100+`      |

## Install

The venv on caribou is `.severity`. From the project root:

```bash
source .severity/bin/activate
pip install vllm
```

Requires:
- Python 3.11 (matches the venv)
- PyTorch 2.4+ (the venv has 2.12+cu130, well above)
- CUDA 12.x or 13.0 (we are on 13.0)

If the install pulls a torch wheel that conflicts with the existing
2.12+cu130, force-pin first:

```bash
pip install --no-deps vllm
pip install -r .severity/requirements-vllm.txt   # generate via pip freeze pre-install
```

Validate the install:

```bash
python -c "from vllm import LLM, SamplingParams; print('vllm OK')"
```

## Usage

### Run a single (model, dataset) pair with vLLM

```bash
PYTHONPATH=src python -m experiments.evaluate_models \
    --dataset medqa --model qwen3-14b \
    --limit 100 --gpu 0 --backend vllm
```

### Run the smoke matrix with vLLM

```bash
./experiments/run_local_smoke_parallel.sh \
    --gpus 0,1,2 --limit 100 --backend vllm
```

### Benchmark HF vs vLLM on one model

```bash
# HF baseline
PYTHONPATH=src python -m experiments.bench_inference \
    --model qwen3-14b --backend hf

# vLLM
PYTHONPATH=src python -m experiments.bench_inference \
    --model qwen3-14b --backend vllm

# Side-by-side
PYTHONPATH=src python -m experiments.bench_inference --compare \
    experiments/benchmarks/*_qwen3-14b_*hf*.json \
    experiments/benchmarks/*_qwen3-14b_*vllm*.json
```

## Known limitations

1. **Engine lifetime**: vLLM claims a fixed fraction of GPU memory at
   load. We destroy the previous engine before loading a new one.
   Loading 11 datasets x 10 models = 10 engine loads (not 110), since
   the same engine is reused across all datasets of a given model.

2. **bnb-4bit checkpoint compatibility**: some `-unsloth-bnb-4bit`
   checkpoints carry config fields that vLLM's bnb loader rejects with
   `ValueError`. The loader falls back to the standard `-bnb-4bit`
   variant automatically (same convention as the HF path).

3. **No assisted_decoding**: vLLM has its own speculative-decoding
   stack (`--speculative-model` on the server, `SpecDecodeWorker` in
   the engine). The HF `assistant_model` knob is not wired up here; if
   you need spec decoding under vLLM, switch to the server mode and
   point it at Qwen3-0.6B.

4. **CUDA 13.2 warning**: Unsloth's docs note that CUDA 13.2 produces
   gibberish outputs. We run CUDA 13.0, so this does not apply to our
   setup -- but if anyone bumps the toolkit, watch for it.

## Falling back to HF

If vLLM crashes on a model (e.g. unsupported architecture, OOM on
PagedAttention warmup), drop the `--backend vllm` flag. The HF path
is the default and is known-stable on every checkpoint in the matrix.
