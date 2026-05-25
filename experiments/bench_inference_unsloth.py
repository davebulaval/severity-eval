"""Standalone unsloth + HF pipeline benchmark.

Mirrors `experiments/bench_inference.py` but runs the **legacy** unsloth +
transformers.pipeline path (removed from main in the vllm-only refactor).
Intended to be invoked from a venv where unsloth is installed alongside
torch + transformers + bitsandbytes, *without* vLLM.

Usage:
    PYTHONPATH=src python -m experiments.bench_inference_unsloth \\
        --model qwen3-14b --n-samples 8 --gpu 0

Output: experiments/benchmarks/<branch>_<commit>_unsloth_<model>_<ts>.json

The implementation is intentionally a self-contained replica of the
pre-refactor _evaluate_local path, with use_cache=False (the safety net
for the unsloth fast-forward RoPE broadcast bug on SWA architectures).
The PROMPTS list is shared with bench_inference.py via duck-import; if
that import fails (when running outside the main repo), we fall back to
a hard-coded mini set.

To compare with vLLM, run the vLLM-side bench and then
`bench_inference.py --compare`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

log = logging.getLogger("bench-unsloth")

BENCH_DIR = Path("experiments/benchmarks")

try:
    from experiments.bench_inference import PROMPTS
except ImportError:
    PROMPTS = [
        "A patient presents with chest pain. Likely diagnosis? Answer A, B, C, D.",
        "FY2023 revenue $4521M, FY2022 $3890M. Growth in percentage points?",
        "Explain in three sentences why frequency-severity is better than accuracy.",
    ]


_LOCAL_MODELS = {
    "llama-3.3-70b": "unsloth/Llama-3.3-70B-Instruct-unsloth-bnb-4bit",
    "deepseek-r1-distill-70b": "unsloth/DeepSeek-R1-Distill-Llama-70B-unsloth-bnb-4bit",
    "qwq-32b": "unsloth/QwQ-32B-unsloth-bnb-4bit",
    "qwen3-30b-a3b": "unsloth/Qwen3-30B-A3B-unsloth-bnb-4bit",
    "gemma-2-27b": "unsloth/gemma-2-27b-it-unsloth-bnb-4bit",
    "mistral-small-3": "unsloth/Mistral-Small-3.1-24B-Instruct-2503-unsloth-bnb-4bit",
    "qwen3-14b": "unsloth/Qwen3-14B-unsloth-bnb-4bit",
    "phi-4": "unsloth/phi-4-unsloth-bnb-4bit",
    "gemma-2-9b": "unsloth/gemma-2-9b-it-unsloth-bnb-4bit",
    "granite-3.2-8b": "unsloth/granite-3.2-8b-instruct-unsloth-bnb-4bit",
}

_THINKING = ("qwen3", "qwq", "deepseek-r1-distill")


def _git(*args: str) -> str:
    try:
        return (
            subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _peak_vram_gb() -> float:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / (1024**3)
    except ImportError:
        return 0.0


def _reset_vram_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def run_benchmark(model_name: str, n_samples: int, max_new_tokens: int) -> dict:
    """Load model via unsloth, run prompts via HF pipeline (use_cache=False)."""
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if model_name not in _LOCAL_MODELS:
        raise SystemExit(
            f"Unknown model '{model_name}'. Known: {sorted(_LOCAL_MODELS.keys())}"
        )
    model_id = _LOCAL_MODELS[model_name]

    is_thinking = any(p in model_id.lower() for p in _THINKING)
    if is_thinking:
        max_new_tokens = min(max_new_tokens * 16, 2048)

    log.info("Loading unsloth model %s (n_samples=%d)", model_id, n_samples)

    import torch
    from transformers import pipeline
    from unsloth import FastLanguageModel

    t0 = time.perf_counter()
    seq_len = 8192 if "gemma-2" in model_id.lower() else 32768
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_id,
        max_seq_length=seq_len,
        device_map="auto",
        max_memory={0: "45GiB"},
        dtype=None,
        load_in_4bit=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    try:
        model.config._attn_implementation = "eager"
    except AttributeError:
        pass
    try:
        model.generation_config.cache_implementation = "dynamic"
    except AttributeError:
        pass
    load_s = time.perf_counter() - t0
    log.info("Loaded in %.1fs", load_s)

    gen = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=1,
        dtype="float16",
        return_full_text=False,
        max_new_tokens=max_new_tokens,
        padding=True,
        truncation=False,
        do_sample=False,
        use_cache=False,
    )

    prompts = (PROMPTS * ((n_samples // len(PROMPTS)) + 1))[:n_samples]
    _reset_vram_peak()

    latency: list[float] = []
    out_tokens: list[int] = []
    for i, prompt in enumerate(prompts):
        t_start = time.perf_counter()
        with torch.no_grad():
            outputs = gen(prompt)
        latency.append(time.perf_counter() - t_start)
        text = outputs[0]["generated_text"] if outputs else ""
        out_tokens.append(len(tokenizer.encode(text, add_special_tokens=False)))
        log.info(
            "  sample %d/%d: %.2fs, %d tokens",
            i + 1,
            n_samples,
            latency[-1],
            out_tokens[-1],
        )

    total_elapsed = sum(latency)
    total_out_tokens = sum(out_tokens)
    return {
        "model_name": model_name,
        "model_id": model_id,
        "backend": "unsloth+hf",
        "attn_impl": "eager",
        "thinking": is_thinking,
        "n_samples": n_samples,
        "max_new_tokens": max_new_tokens,
        "load_seconds": round(load_s, 2),
        "total_seconds": round(total_elapsed, 2),
        "mean_latency_seconds": round(statistics.mean(latency), 2),
        "p50_latency_seconds": round(statistics.median(latency), 2),
        "p90_latency_seconds": round(
            statistics.quantiles(latency, n=10)[8]
            if len(latency) >= 10
            else max(latency),
            2,
        ),
        "total_generated_tokens": total_out_tokens,
        "tokens_per_second": round(total_out_tokens / total_elapsed, 1)
        if total_elapsed > 0
        else 0.0,
        "peak_vram_gb": round(_peak_vram_gb(), 2),
        "per_sample_latency_seconds": [round(s, 2) for s in latency],
        "per_sample_out_tokens": out_tokens,
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_commit": _git("rev-parse", "--short=8", "HEAD"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _save(metrics: dict) -> Path:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    name = (
        f"{metrics['git_branch'].replace('/', '_')}_"
        f"{metrics['git_commit']}_"
        f"unsloth_"  # explicit tag so compare picks the right pair
        f"{metrics['model_name']}_"
        f"{metrics['timestamp_utc'].replace(':', '-')}.json"
    )
    path = BENCH_DIR / name
    path.write_text(json.dumps(metrics, indent=2))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    metrics = run_benchmark(args.model, args.n_samples, args.max_new_tokens)
    path = _save(metrics)
    log.info("Saved benchmark to %s", path)

    print(
        f"\n--- unsloth+hf  {metrics['model_name']}  commit {metrics['git_commit']} ---"
    )
    print(f"  load_seconds       : {metrics['load_seconds']}")
    print(f"  total_seconds      : {metrics['total_seconds']}")
    print(f"  mean_latency_s     : {metrics['mean_latency_seconds']}")
    print(f"  tokens_per_second  : {metrics['tokens_per_second']}")
    print(f"  peak_vram_gb       : {metrics['peak_vram_gb']}")


if __name__ == "__main__":
    main()
