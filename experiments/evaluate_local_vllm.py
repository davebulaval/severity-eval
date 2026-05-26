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
from pathlib import Path
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
# Caps reflect each checkpoint's native max_position_embeddings. CUAD
# asks for ~33 K -- families capped below that (Qwen2.5, Qwen3, QwQ at
# 32 K; phi-4 at 16 K) need this entry so vLLM does not reject the init,
# and prompts get tail-truncated by the caller. gemma-3 (12B/27B), the
# gpt-oss models, mistral-small-3, llama-3.3-70b, deepseek-r1-distill,
# and granite-3.2-8b all handle >= 33 K natively.
_MODEL_MAX_LEN_CAPS: tuple[tuple[str, int], ...] = (
    ("phi-4", 16384),
    ("qwen2.5", 32768),
    ("qwen3", 32768),
    ("qwq", 32768),
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
    output_path: Path | None = None,
    chunk_size: int = 100,
    force: bool = False,
) -> list[dict]:
    """Batch inference for local models via vLLM.

    Mirrors the contract of experiments.evaluate_models._evaluate_local
    but uses vLLM under the hood. Returns a list of dicts ready to be
    appended to the results JSON.

    Checkpointing behavior (when output_path is not None):
      * Submits the work in chunks of ``chunk_size`` prompts and rewrites
        the output JSON after each chunk; a crash at instance 350 of 1000
        leaves a usable file with the first 300 results.
      * On startup, if ``output_path`` already contains results, those
        entries are kept and only the rows with new ``id`` values are
        run through vLLM. This is how you extend a run from --limit 1000
        to --limit 2000 without re-doing the first 1000 (pass force=False).
      * ``force=True`` ignores any existing file and re-runs everything.
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

    # ---- Resume from existing output_path (checkpoint / extend) ----
    completed_results: list[dict] = []
    completed_ids: set[str] = set()
    if output_path is not None and output_path.exists() and not force:
        try:
            import json as _json

            completed_results = _json.loads(output_path.read_text())
            if not isinstance(completed_results, list):
                raise ValueError("output_path is not a JSON list")
            completed_ids = {str(r["id"]) for r in completed_results if "id" in r}
            log.info(
                "Resuming from %d completed entries in %s",
                len(completed_results),
                output_path.name,
            )
        except Exception as exc:
            log.warning(
                "Could not read existing output_path %s (%s); starting fresh",
                output_path,
                exc,
            )
            completed_results = []
            completed_ids = set()

    if completed_ids:
        df_pending = df[~df["id"].astype(str).isin(completed_ids)].reset_index(
            drop=True
        )
    else:
        df_pending = df.reset_index(drop=True)

    if len(df_pending) == 0:
        log.info(
            "All %d items already complete in %s; nothing to do",
            len(df),
            output_path.name if output_path else "<no path>",
        )
        return completed_results

    if completed_ids:
        log.info(
            "%d new items to evaluate (resuming after %d already done)",
            len(df_pending),
            len(completed_ids),
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
    for _, row in df_pending.iterrows():
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

    # Pre-tokenize, truncate, and decode back to text. We used to pass
    # the token IDs directly via the `prompt_token_ids=` kwarg, but some
    # vLLM builds reject that kwarg with "LLM.generate() got an
    # unexpected keyword argument 'prompt_token_ids'". The round-trip
    # encode/decode is safe for our use case (instruction-tuned models
    # whose tokenizers handle decode-then-encode idempotently for normal
    # text), and lets the chunk loop below use a single uniform call
    # signature `llm.generate(text_prompts, sampling)`.
    if needs_truncation:
        truncated_prompts: list[str] = []
        for p in prompts:
            ids = tokenizer.encode(p, add_special_tokens=False)
            if len(ids) > prompt_token_budget:
                ids = ids[-prompt_token_budget:]
                truncated_prompts.append(
                    tokenizer.decode(ids, skip_special_tokens=False)
                )
            else:
                truncated_prompts.append(p)
        prompts = truncated_prompts

    # ---- Chunked submission with checkpoint saves after every chunk ----
    # vLLM's continuous batching is fastest with a large batch, but a
    # single crash mid-batch loses everything. We submit `chunk_size`
    # prompts at a time and rewrite output_path atomically after each
    # chunk so a SIGINT at instance 350/1000 still leaves the first 300
    # results on disk. chunk_size=100 keeps the overhead low (one engine
    # call per 100 prompts) while bounding the worst-case loss.
    all_results: list[dict] = list(completed_results)
    n_chunks = (len(prompts) + chunk_size - 1) // chunk_size
    log.info(
        "Submitting %d prompts to vLLM in %d chunk(s) of <=%d "
        "(max_new_tokens=%d, max_model_len=%d, needs_truncation=%s)",
        len(prompts),
        n_chunks,
        chunk_size,
        max_new_tokens,
        max_model_len,
        needs_truncation,
    )

    overall_t0 = time.perf_counter()
    for chunk_idx, chunk_start in enumerate(range(0, len(prompts), chunk_size)):
        chunk_end = min(chunk_start + chunk_size, len(prompts))
        chunk_rows = rows[chunk_start:chunk_end]
        chunk_options = options_per_row[chunk_start:chunk_end]

        t0 = time.perf_counter()
        raw_outputs = llm.generate(prompts[chunk_start:chunk_end], sampling)
        n_chunk = chunk_end - chunk_start
        elapsed = time.perf_counter() - t0
        log.info(
            "  chunk %d/%d : %d prompts in %.1fs (%.1f prompts/s)",
            chunk_idx + 1,
            n_chunks,
            n_chunk,
            elapsed,
            n_chunk / max(elapsed, 1e-6),
        )

        # `outputs[i]` corresponds to `prompts[i]` per the vLLM API contract.
        for row, output, options in zip(
            chunk_rows, raw_outputs, chunk_options, strict=True
        ):
            text = output.outputs[0].text.strip() if output.outputs else ""
            prediction = _strip_think_tags(text) if thinking else text
            scoring = score_prediction(
                prediction,
                str(row["answer"]),
                options=options,
            )
            all_results.append(
                {
                    **row.to_dict(),
                    "model": model_name,
                    "prediction": prediction,
                    "correct": scoring["correct"],
                    "score_method": scoring["score_method"],
                }
            )

        if output_path is not None:
            _atomic_write_json(output_path, all_results)
            log.info(
                "  checkpoint: %d/%d total saved to %s",
                len(all_results),
                len(completed_results) + len(prompts),
                output_path.name,
            )

    overall_elapsed = time.perf_counter() - overall_t0
    log.info(
        "vLLM done : %d new prompts in %.1fs (%.1f prompts/s overall)",
        len(prompts),
        overall_elapsed,
        len(prompts) / max(overall_elapsed, 1e-6),
    )
    return all_results


def _atomic_write_json(path: Path, payload: list[dict]) -> None:
    """Write JSON via a tmp file + rename so a crash mid-write never leaves
    a half-written checkpoint that breaks future resumes.
    """
    import json as _json

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(_json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    tmp.replace(path)
