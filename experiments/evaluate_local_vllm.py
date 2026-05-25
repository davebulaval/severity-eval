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


# Per-model hard caps on max_model_len. These come from the model's
# `max_position_embeddings` in config.json -- training-time limits
# that the model simply has not learned positions beyond. CUAD asks
# for ~33 K tokens which exceeds phi-4 (16 K); the caller truncates
# prompts to (cap - max_new_tokens) via SamplingParams.
#
# gemma-3 (12B, 27B) is NOT capped: it natively supports 128 K context
# (the gemma-2 entries were removed when we swapped to gemma-3).
# gpt-oss-20b / gpt-oss-120b also handle 128 K natively.
# llama-3.3-70b at TP=2 has enough KV cache budget for 33 K.
_MODEL_MAX_LEN_CAPS: tuple[tuple[str, int], ...] = (
    ("phi-4", 16384),  # native max_position_embeddings
)


def _max_model_len_for(model_id: str, default: int = 8192) -> int:
    """Pick the max sequence length vLLM is allowed to instantiate for
    this model.

    Returns the per-model cap from `_MODEL_MAX_LEN_CAPS` (matched on a
    substring of `model_id`, case-insensitive) if there is one, else
    `default`. Callers should always pass this through min(requested,
    cap) -- the cap is a hard ceiling, not a default.
    """
    lower = model_id.lower()
    for pattern, cap in _MODEL_MAX_LEN_CAPS:
        if pattern in lower:
            return cap
    return default


def _quantization_for(model_id: str) -> str | None:
    """Pick vLLM's quantization arg from the model_id suffix.

    Returns:
      - "bitsandbytes" for Unsloth bnb checkpoints (matched on
        "-bnb-4bit" in the id). These cannot do TP -- _load_local_vllm
        force-caps TP=1 when this branch fires.
      - "awq_marlin" / "gptq_marlin" / "compressed-tensors" / "fp8"
        for ids whose suffix matches the corresponding quantization.
      - None as the default. vLLM 0.9+ reads quant_config from the
        model's config.json when quantization=None and picks the
        right kernel (covers FP16, MXFP4 native, and anything else
        without a recognizable suffix).
    """
    lower = model_id.lower()
    if "-bnb-4bit" in lower:
        return "bitsandbytes"
    if lower.endswith("-awq") or "-awq-" in lower or lower.endswith("-awq-int4"):
        return "awq_marlin"
    if lower.endswith("-gptq") or "-gptq-" in lower:
        return "gptq_marlin"
    if "w4a16" in lower or "compressed-tensors" in lower:
        return "compressed-tensors"
    # FP8 packaging varies by publisher: Qwen ships native fp8, RedHat
    # ships compressed-tensors with fp8 weights. Returning None lets
    # vLLM read quant_method from config.json and pick the right kernel.
    if lower.endswith("-fp8") or "-fp8-" in lower or "-fp8-dynamic" in lower:
        return None
    return None


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
    tensor_parallel_size: int | None = None,
):
    """Load (or return cached) vLLM engine for model_id.

    On a fresh model_id, or when the requested max_model_len exceeds the
    cached engine's max_model_len, the previous engine is destroyed and
    a new one is loaded.

    `tensor_parallel_size` defaults to 1 (single-GPU). When None, the
    env var SEVERITY_EVAL_TP is consulted (so the CLI flag in
    evaluate_models.py can reach this helper without threading the arg
    through the public evaluate_local_vllm signature). Pass > 1 to
    shard the model across that many GPUs.

    Raises ValueError if gpu_memory_utilization is outside (0, 1].
    """
    import os as _os

    if tensor_parallel_size is None:
        try:
            tensor_parallel_size = int(_os.environ.get("SEVERITY_EVAL_TP", "1"))
        except ValueError:
            tensor_parallel_size = 1
    if tensor_parallel_size < 1:
        raise ValueError(
            f"tensor_parallel_size must be >= 1, got {tensor_parallel_size}"
        )

    # vLLM raises ValueError("Prequant BitsAndBytes models with tensor
    # parallelism is not supported") at engine init time for any
    # bitsandbytes checkpoint with TP>1. Cap silently so the wrapper
    # script can pass the same TP for every model without per-model
    # conditionals. In the current MODELS table only granite-3.2-8b
    # lacks an AWQ/GPTQ/FP8/w4a16 equivalent; it will run single-GPU
    # and vLLM still parallelizes within the single GPU.
    if _quantization_for(model_id) == "bitsandbytes" and tensor_parallel_size > 1:
        log.warning(
            "bitsandbytes checkpoint %s does not support TP>1; "
            "capping tensor_parallel_size from %d to 1",
            model_id,
            tensor_parallel_size,
        )
        tensor_parallel_size = 1

    if not (0 < gpu_memory_utilization <= 1.0):
        raise ValueError(
            f"gpu_memory_utilization must be in (0, 1], got {gpu_memory_utilization}"
        )

    requested_max_len = max_model_len or _max_model_len_for(model_id)

    # Reuse the cached engine only if it matches model_id AND has at
    # least the requested context length AND was built with the same
    # tensor_parallel_size. CUAD asks for ~33 K while MedQA needs 4 K;
    # without the max_len check the smaller engine would crash on the
    # long prompt. The TP check matters when phase 1 (TP=1, per-model
    # parallel) and phase 2 (TP=3, sharded) are interleaved in the same
    # Python process -- without it the cache would hand back a TP=1
    # engine for a TP=3 request.
    cached_model_id = _vllm_engine.get("model_id")
    cached_max_len = _vllm_engine.get("max_model_len", 0)
    cached_tp = _vllm_engine.get("tensor_parallel_size", 1)
    if (
        cached_model_id == model_id
        and cached_max_len >= requested_max_len
        and cached_tp == tensor_parallel_size
    ):
        return _vllm_engine["llm"]

    _destroy_engine()
    max_model_len = requested_max_len

    # Defer the vllm import until we actually need to construct an engine.
    # Validation + cache lookup above must run even in environments where
    # vllm is not installed (e.g. unit tests).
    from vllm import LLM

    def _try_load(repo_id: str):
        quantization = _quantization_for(repo_id)
        log.info(
            "Loading vLLM engine %s (quantization=%s, max_model_len=%d, "
            "gpu_util=%.2f, tp=%d) ...",
            repo_id,
            quantization,
            max_model_len,
            gpu_memory_utilization,
            tensor_parallel_size,
        )
        return LLM(
            model=repo_id,
            quantization=quantization,
            dtype="bfloat16",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
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
    _vllm_engine["tensor_parallel_size"] = tensor_parallel_size
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

    # vLLM needs max_model_len >= max_length + max_new_tokens. When the
    # dataset (CUAD ~33 K) exceeds the model's hard cap (only phi-4 at
    # 16 K in the current MODELS table -- gemma-3 has 128 K native),
    # we instantiate at the cap and let
    # SamplingParams.truncate_prompt_tokens drop the leading tokens so
    # the trailing portion (which holds the question + answer choices)
    # fits. We keep the largest tail possible, max_model_len -
    # max_new_tokens. Loss of leading context is documented as a
    # limitation when reporting CUAD numbers on phi-4.
    requested_max_len = max(max_length + max_new_tokens, 4096)
    model_cap = _max_model_len_for(model_id, default=requested_max_len)
    max_model_len = min(requested_max_len, model_cap)
    truncate_to = max_model_len - max_new_tokens
    if max_model_len < requested_max_len:
        log.warning(
            "Capping max_model_len from %d to %d for %s on %s; "
            "prompts above %d tokens will keep only the last %d.",
            requested_max_len,
            max_model_len,
            model_id,
            dataset_name,
            truncate_to,
            truncate_to,
        )

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

    # Truncate manually via tokenizer when needed. We used to pass
    # SamplingParams.truncate_prompt_tokens, but some vLLM versions
    # reject that kwarg ("Unexpected keyword argument
    # 'truncate_prompt_tokens'"), so we do the truncation here once.
    # Only the trailing portion of the prompt is kept; on CUAD this
    # means the contract body is cut and the question+choices remain
    # (documented limitation on phi-4 only -- gemma-3 and the others
    # all support the full 33 K context).
    prompt_token_budget = max(truncate_to, 1)
    needs_truncation = False
    for p in prompts:
        if len(tokenizer.encode(p, add_special_tokens=False)) > prompt_token_budget:
            needs_truncation = True
            break

    log.info(
        "Submitting %d prompts to vLLM (max_new_tokens=%d, max_model_len=%d, "
        "needs_truncation=%s)",
        len(prompts),
        max_new_tokens,
        max_model_len,
        needs_truncation,
    )

    t0 = time.perf_counter()
    if needs_truncation:
        token_id_lists: list[list[int]] = []
        for p in prompts:
            ids = tokenizer.encode(p, add_special_tokens=False)
            if len(ids) > prompt_token_budget:
                ids = ids[-prompt_token_budget:]
            token_id_lists.append(ids)
        raw_outputs = llm.generate(
            prompt_token_ids=token_id_lists, sampling_params=sampling
        )
    else:
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
