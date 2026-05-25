"""vLLM backend for local-model evaluation.

vLLM provides continuous batching, PagedAttention KV cache, and prefix
caching. It is the only local-inference engine in this repo. An earlier
HF + transformers.pipeline path existed but was removed -- vLLM is
faster, more stable, and avoids the cache/JIT bugs we hit on that
stack.

Public surface:

    - _load_local_vllm(model_id, ...)  ->  vllm.LLM
    - evaluate_local_vllm(df, model_name, ...) -> list[dict]

experiments.evaluate_models.evaluate_model dispatches to evaluate_local_vllm
when provider == "local".

Install (already pinned in requirements.txt):
    pip install vllm  # requires CUDA 12.x or 13.0 + recent torch

Known limitations:
    - Loading a vLLM engine claims a fixed fraction of GPU memory; we
      destroy the previous engine before loading a new one.
    - assisted_decoding / draft models are not wired up here; vLLM has
      its own speculative-decoding stack reachable via the
      --speculative-model server flag.
    - Some Unsloth Dynamic 2.0 (-unsloth-bnb-4bit) checkpoints carry
      config that vLLM's bnb loader rejects; we fall back to the
      standard -bnb-4bit HF repo on load failure. The "-unsloth-"
      suffix only refers to the upstream HF repository name (i.e.
      Unsloth-published Dynamic 2.0 quants), not to the unsloth
      Python library, which is not installed in this venv.
"""

from __future__ import annotations

import gc
import logging
import time
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

# Cache: one engine at a time. Switching engines destroys the previous.
_vllm_engine: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Engine loading
# ---------------------------------------------------------------------------


def _max_model_len_for(model_id: str, default: int = 8192) -> int:
    """Pick a sensible max_model_len. gemma-2 caps at 8K natively."""
    if "gemma-2" in model_id.lower():
        return 8192
    return default


def _destroy_engine() -> None:
    """Drop the cached engine and free its GPU memory."""
    if "llm" not in _vllm_engine:
        return
    log.info("Destroying previous vLLM engine")
    try:
        from vllm.distributed import destroy_distributed_environment

        destroy_distributed_environment()
    except (ImportError, RuntimeError) as exc:
        log.debug("destroy_distributed_environment skipped: %s", exc)
    _vllm_engine.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _load_local_vllm(
    model_id: str,
    *,
    max_model_len: int | None = None,
    gpu_memory_utilization: float = 0.92,
):
    """Load (or return cached) vLLM engine for model_id.

    On a fresh model_id, or when the requested max_model_len exceeds the
    cached engine's max_model_len, the previous engine is destroyed and
    a new one is loaded.

    Raises ValueError if gpu_memory_utilization is outside (0, 1].
    """
    if not (0 < gpu_memory_utilization <= 1.0):
        raise ValueError(
            f"gpu_memory_utilization must be in (0, 1], got {gpu_memory_utilization}"
        )

    requested_max_len = max_model_len or _max_model_len_for(model_id)

    # Reuse the cached engine only if it matches both model_id AND has
    # at least the requested context length. CUAD asks for ~33 K while
    # MedQA needs 4 K; without this check the smaller engine would
    # crash on the long prompt.
    cached_model_id = _vllm_engine.get("model_id")
    cached_max_len = _vllm_engine.get("max_model_len", 0)
    if cached_model_id == model_id and cached_max_len >= requested_max_len:
        return _vllm_engine["llm"]

    _destroy_engine()
    max_model_len = requested_max_len

    # Defer the vllm import until we actually need to construct an engine.
    # Validation + cache lookup above must run even in environments where
    # vllm is not installed (e.g. unit tests).
    from vllm import LLM

    def _try_load(repo_id: str):
        log.info(
            "Loading vLLM engine %s (max_model_len=%d, gpu_util=%.2f) ...",
            repo_id,
            max_model_len,
            gpu_memory_utilization,
        )
        return LLM(
            model=repo_id,
            quantization="bitsandbytes",
            dtype="bfloat16",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=True,
            enforce_eager=False,
            trust_remote_code=True,
        )

    # Why catch RuntimeError + AssertionError too: when an Unsloth Dynamic 2.0
    # checkpoint has tensor shapes vLLM's bnb loader rejects (e.g.
    # DeepSeek-R1-Distill-Llama-70B-unsloth-bnb-4bit), the failure happens in
    # the EngineCore sub-process as an AssertionError on
    # linear.py weight_loader; the parent sees it wrapped as
    # "Engine core initialization failed" (RuntimeError). We surface both so
    # the fallback to the non-Dynamic -bnb-4bit repo kicks in.
    try:
        llm = _try_load(model_id)
    except (OSError, ValueError, RuntimeError, AssertionError) as exc:
        if "unsloth-bnb-4bit" not in model_id:
            raise
        fallback = model_id.replace("unsloth-bnb-4bit", "bnb-4bit")
        log.warning(
            "Dynamic variant %s rejected by vLLM (%s); falling back to %s",
            model_id,
            type(exc).__name__,
            fallback,
        )
        llm = _try_load(fallback)
        model_id = fallback

    _vllm_engine["model_id"] = model_id
    _vllm_engine["max_model_len"] = max_model_len
    _vllm_engine["llm"] = llm
    return llm


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_local_vllm(
    df: pd.DataFrame,
    model_name: str,
    model_id: str,
    dataset_name: str,
    prompt_style: str,
) -> list[dict]:
    """Batch inference for local models via vLLM.

    Mirrors the contract of experiments.evaluate_models._evaluate_local
    but uses vLLM under the hood. Returns a list of dicts ready to be
    appended to the results JSON.
    """
    # Import lazily so the HF path does not pay the vLLM import cost
    # (it pulls torch, transformers, multiple CUDA libs).
    from vllm import SamplingParams

    from experiments.evaluate_models import (
        DATASET_INFERENCE_CONFIG,
        _DEFAULT_INFERENCE_CONFIG,
        _THINKING_MAX_NEW_TOKENS_CAP,
        _THINKING_TOKEN_MULTIPLIER,
        _build_prompt_for_row,
        _is_thinking_model,
        _strip_think_tags,
        score_prediction,
    )

    max_new_tokens, max_length = DATASET_INFERENCE_CONFIG.get(
        dataset_name, _DEFAULT_INFERENCE_CONFIG
    )
    thinking = _is_thinking_model(model_id)
    if thinking:
        scaled = max_new_tokens * _THINKING_TOKEN_MULTIPLIER
        max_new_tokens = min(scaled, _THINKING_MAX_NEW_TOKENS_CAP)
        log.info(
            "Thinking model — max_new_tokens=%d (cap=%d)",
            max_new_tokens,
            _THINKING_MAX_NEW_TOKENS_CAP,
        )

    # vLLM needs max_model_len >= max_length + max_new_tokens
    max_model_len = max(max_length + max_new_tokens, 4096)
    llm = _load_local_vllm(model_id, max_model_len=max_model_len)
    tokenizer = llm.get_tokenizer()

    # NOTE on top_p: we used to pass top_p=1.0 but vLLM's
    # topk_topp_sampler.forward_cuda routes to flashinfer_sample whenever
    # k or p is set, regardless of value. flashinfer's JIT compile path
    # is broken on this venv's CUDA toolkit (nvcc/ptxas PTX-version
    # mismatch), so we explicitly leave top_p unset to force the native
    # PyTorch sampler. Combined with VLLM_USE_FLASHINFER_SAMPLER=0, this
    # is belt-and-braces against the same crash.
    sampling = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
    )

    # Build prompts. We use the tokenizer's chat template when available
    # so the same prompt yields the same predictions as the HF path.
    prompts: list[str] = []
    options_per_row: list[Any] = []
    rows: list[Any] = []
    for _, row in df.iterrows():
        raw_prompt, options = _build_prompt_for_row(row, dataset_name, prompt_style)
        # Some tokenizers lack a chat_template and raise ValueError, others
        # are missing the apply method (older tokenizers). Anything beyond
        # those two should surface so we don't silently send malformed
        # prompts.
        try:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": raw_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except (AttributeError, ValueError, KeyError, TypeError) as exc:
            log.warning(
                "apply_chat_template failed (%s); using raw prompt", type(exc).__name__
            )
            formatted = raw_prompt
        prompts.append(formatted)
        options_per_row.append(options)
        rows.append(row)

    log.info(
        "Submitting %d prompts to vLLM (max_new_tokens=%d, max_model_len=%d)",
        len(prompts),
        max_new_tokens,
        max_model_len,
    )

    t0 = time.perf_counter()
    raw_outputs = llm.generate(prompts, sampling)
    elapsed = time.perf_counter() - t0
    log.info(
        "vLLM %d prompts in %.1fs (%.1f prompts/s)",
        len(prompts),
        elapsed,
        len(prompts) / max(elapsed, 1e-6),
    )

    # vLLM may return outputs in submission order or by completion;
    # `outputs[i]` corresponds to `prompts[i]` per the API contract.
    results: list[dict] = []
    for row, output, options in zip(rows, raw_outputs, options_per_row, strict=True):
        text = output.outputs[0].text.strip() if output.outputs else ""
        prediction = _strip_think_tags(text) if thinking else text
        scoring = score_prediction(
            prediction,
            str(row["answer"]),
            options=options,
        )
        results.append(
            {
                **row.to_dict(),
                "model": model_name,
                "prediction": prediction,
                "correct": scoring["correct"],
                "score_method": scoring["score_method"],
            }
        )
    return results
