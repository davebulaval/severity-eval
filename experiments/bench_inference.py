"""Inference benchmark.

Measures wall-clock and tokens/sec for a single local model on a fixed
set of prompts. Designed to compare two branches (baseline vs speedup)
without depending on HF dataset downloads or network state.

Usage:

    # Run a benchmark with the current code
    PYTHONPATH=src python -m experiments.bench_inference \\
        --model qwen3-14b --n-samples 8 --gpu 0

    # Output: experiments/benchmarks/<branch>_<commit>_<model>_<ts>.json

    # Compare two prior runs
    PYTHONPATH=src python -m experiments.bench_inference --compare \\
        experiments/benchmarks/main_53d8b97_qwen3-14b_*.json \\
        experiments/benchmarks/speedup_e529394_qwen3-14b_*.json

The benchmark records:
    - per-sample latency (mean, p50, p90)
    - tokens/sec across the batch
    - peak VRAM
    - generation_config snapshot (so we can tell what changed)
    - git branch + commit so two runs cannot be confused
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

# Make sure the experiments package is importable.
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

log = logging.getLogger("bench")

BENCH_DIR = Path("experiments/benchmarks")

# Fixed set of prompts covering the four task types we evaluate in the
# paper. Short and long are mixed so group-by-length has something to
# work with.
PROMPTS: list[str] = [
    # MCQ-style (short prompt, short answer)
    "A patient presents with chest pain radiating to the left arm. "
    "Which is the most likely diagnosis? A) Myocardial infarction "
    "B) Pericarditis C) GERD D) Pulmonary embolism. Answer with the letter only.",
    # Numeric extraction (medium prompt, short answer)
    "In FY2023 the company reported revenue of $4,521 million, up from "
    "$3,890 million the prior year. What was the year-over-year revenue "
    "growth in percentage points? Respond with a single number.",
    # Legal clause classification (medium prompt)
    "Contract clause: 'Either party may terminate this Agreement upon "
    "thirty (30) days written notice to the other party.' Does this "
    "clause grant a unilateral termination right to both parties? "
    "Answer yes or no.",
    # RAG-style insurance question (long prompt)
    "Policy excerpt: 'Coverage applies to direct physical loss or "
    "damage caused by a covered cause of loss. Covered causes of loss "
    "are fire, lightning, explosion, smoke, vehicle, aircraft, riot, "
    "vandalism, and theft. Excluded causes include flood, earthquake, "
    "war, and nuclear hazard.' Question: Is damage from a hurricane "
    "covered by this policy? Justify in one sentence.",
    # Long-form reasoning (medium prompt, longer expected output)
    "Explain in three sentences why the actuarial frequency-severity "
    "decomposition is more informative than raw accuracy when comparing "
    "two language models on a high-stakes domain.",
    # Open-ended QA (short prompt)
    "What is the relationship between Value-at-Risk and Tail Value-at-Risk?",
    # Translation-like (medium prompt)
    "Convert the following sentence to formal English: 'gonna check "
    "the deal and let you know whats up by tmrw'.",
    # Code-like (short prompt)
    "Given a Python list of integers, write a one-line expression that "
    "returns the sum of squares of even numbers. Use a list comprehension.",
]


def _git(*args: str) -> str:
    """Run git and return stdout stripped, or 'unknown' if it fails."""
    try:
        out = subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _peak_vram_gb() -> float:
    """Peak GPU memory used since last reset, in GB."""
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


def _model_id_for(model_name: str) -> str:
    """Resolve a logical model name (e.g. 'qwen3-14b') to its HF id
    by reading the MODELS dict from evaluate_models."""
    from experiments.evaluate_models import MODELS

    if model_name not in MODELS:
        raise SystemExit(
            f"Unknown model '{model_name}'. Known: {sorted(MODELS.keys())}"
        )
    spec = MODELS[model_name]
    if spec["provider"] != "local":
        raise SystemExit(
            f"Model '{model_name}' is provider={spec['provider']}; "
            "bench_inference targets only local models."
        )
    return spec["model_id"]


def _run_hf(
    model_name: str, prompts: list[str], max_new_tokens: int
) -> tuple[float, list[float], list[int], str]:
    """HF + unsloth path. Returns (load_seconds, per_sample_latency,
    per_sample_out_tokens, attn_impl)."""
    import torch
    from transformers import pipeline

    from experiments.evaluate_models import _attn_impl_for, _load_local_model

    model_id = _model_id_for(model_name)
    t0 = time.perf_counter()
    model, tokenizer = _load_local_model(model_id)
    load_s = time.perf_counter() - t0

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

    latency: list[float] = []
    out_tokens: list[int] = []
    for prompt in prompts:
        t_start = time.perf_counter()
        with torch.no_grad():
            outputs = gen(prompt)
        latency.append(time.perf_counter() - t_start)
        text = outputs[0]["generated_text"] if outputs else ""
        out_tokens.append(len(tokenizer.encode(text, add_special_tokens=False)))
    return load_s, latency, out_tokens, _attn_impl_for(model_id)


def _run_vllm(
    model_name: str, prompts: list[str], max_new_tokens: int
) -> tuple[float, list[float], list[int], str]:
    """vLLM path. The engine batches internally; we record the batch
    wall-clock and attribute it evenly across prompts for the latency
    columns. Tokens are exact per output."""
    from vllm import SamplingParams

    from experiments.evaluate_local_vllm import _load_local_vllm

    model_id = _model_id_for(model_name)
    max_model_len = max(max_new_tokens + 2048, 4096)
    t0 = time.perf_counter()
    llm = _load_local_vllm(model_id, max_model_len=max_model_len)
    load_s = time.perf_counter() - t0
    tokenizer = llm.get_tokenizer()

    sampling = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
    t_batch = time.perf_counter()
    raw = llm.generate(prompts, sampling)
    batch_elapsed = time.perf_counter() - t_batch

    per_sample = batch_elapsed / max(len(prompts), 1)
    latency = [per_sample] * len(prompts)
    out_tokens = [
        len(tokenizer.encode(o.outputs[0].text, add_special_tokens=False))
        if o.outputs
        else 0
        for o in raw
    ]
    return load_s, latency, out_tokens, "vllm-paged"


def run_benchmark(
    model_name: str,
    n_samples: int,
    max_new_tokens: int,
    backend: str = "hf",
) -> dict:
    """Run the benchmark and return the metrics dict.

    Raises ValueError if n_samples < 1 (the latency aggregations and
    tokens/sec calculations all assume at least one sample).
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    from experiments.evaluate_models import _is_thinking_model

    model_id = _model_id_for(model_name)
    prompts = (PROMPTS * ((n_samples // len(PROMPTS)) + 1))[:n_samples]

    log.info("Loading %s (%s) [backend=%s]", model_name, model_id, backend)
    _reset_vram_peak()
    if backend == "vllm":
        load_s, latency, out_tokens, attn_impl = _run_vllm(
            model_name, prompts, max_new_tokens
        )
    else:
        load_s, latency, out_tokens, attn_impl = _run_hf(
            model_name, prompts, max_new_tokens
        )

    total_elapsed = sum(latency)
    total_out_tokens = sum(out_tokens)
    return {
        "model_name": model_name,
        "model_id": model_id,
        "backend": backend,
        "attn_impl": attn_impl,
        "thinking": _is_thinking_model(model_id),
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
        f"{metrics['model_name']}_"
        f"{metrics['timestamp_utc'].replace(':', '-')}.json"
    )
    path = BENCH_DIR / name
    path.write_text(json.dumps(metrics, indent=2))
    return path


def compare(baseline_path: Path, new_path: Path) -> None:
    """Print a side-by-side comparison of two benchmark JSONs."""
    b = json.loads(Path(baseline_path).read_text())
    n = json.loads(Path(new_path).read_text())
    if b["model_name"] != n["model_name"]:
        log.warning(
            "Comparing different models: %s vs %s",
            b["model_name"],
            n["model_name"],
        )
    keys = [
        ("load_seconds", "lower"),
        ("total_seconds", "lower"),
        ("mean_latency_seconds", "lower"),
        ("p50_latency_seconds", "lower"),
        ("p90_latency_seconds", "lower"),
        ("tokens_per_second", "higher"),
        ("peak_vram_gb", "lower"),
    ]
    print(f"{'metric':<30} {'baseline':>14} {'new':>14} {'delta':>14}")
    print("-" * 76)
    for key, want in keys:
        bv = b.get(key, float("nan"))
        nv = n.get(key, float("nan"))
        delta_pct = (nv - bv) / bv * 100 if bv else float("nan")
        flag = ""
        if want == "lower" and delta_pct < 0:
            flag = "FASTER"
        elif want == "higher" and delta_pct > 0:
            flag = "FASTER"
        elif want == "lower" and delta_pct > 0:
            flag = "SLOWER"
        elif want == "higher" and delta_pct < 0:
            flag = "SLOWER"
        print(f"{key:<30} {bv:>14.2f} {nv:>14.2f} {delta_pct:>+12.1f}%  {flag}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Logical model name (e.g. qwen3-14b)")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument(
        "--backend",
        choices=("hf", "vllm"),
        default="hf",
        help="Inference engine. 'hf' = unsloth + transformers pipeline "
        "(no extra install). 'vllm' = vLLM (pip install vllm).",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE", "NEW"),
        help="Compare two benchmark JSON files",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    if args.compare:
        compare(Path(args.compare[0]), Path(args.compare[1]))
        return

    if not args.model:
        parser.error("--model required when not using --compare")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    metrics = run_benchmark(
        args.model, args.n_samples, args.max_new_tokens, backend=args.backend
    )
    path = _save(metrics)
    log.info("Saved benchmark to %s", path)

    # Pretty print headline numbers
    print(f"\n--- {metrics['model_name']} on commit {metrics['git_commit']} ---")
    print(f"  backend            : {metrics['backend']}")
    print(f"  load_seconds       : {metrics['load_seconds']}")
    print(f"  total_seconds      : {metrics['total_seconds']}")
    print(f"  mean_latency_s     : {metrics['mean_latency_seconds']}")
    print(f"  tokens_per_second  : {metrics['tokens_per_second']}")
    print(f"  peak_vram_gb       : {metrics['peak_vram_gb']}")
    print(f"  attn_impl          : {metrics['attn_impl']}")


if __name__ == "__main__":
    main()
