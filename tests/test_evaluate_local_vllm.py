"""Tests for the pure-logic helpers in experiments.evaluate_local_vllm.

The actual engine loading and inference are not unit tested because they
require vLLM + a GPU. We cover:
    - _max_model_len_for: model-specific context caps
    - _destroy_engine: idempotent cleanup of the global cache
"""

from __future__ import annotations

from unittest.mock import patch

import experiments.evaluate_local_vllm as vllm_mod
from experiments.evaluate_local_vllm import (
    _destroy_engine,
    _max_model_len_for,
    _quantization_for,
)


# ----------------------------------------------------------------------
# _max_model_len_for
# ----------------------------------------------------------------------


def test_max_model_len_does_not_cap_gemma_3():
    """gemma-3 (replacement for gemma-2) natively supports 128K context.
    Verifying it is NOT in the cap table -- so it loads CUAD 33K
    without truncation.
    """
    assert (
        _max_model_len_for("ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g", default=131072)
        == 131072
    )


def test_max_model_len_non_capped_uses_default():
    """A model without an entry in _MODEL_MAX_LEN_CAPS uses the caller's
    default. qwq-32b is not in the cap table.
    """
    assert (
        _max_model_len_for("unsloth/QwQ-32B-unsloth-bnb-4bit", default=32768) == 32768
    )


def test_max_model_len_caps_phi_4_at_16k():
    """phi-4's max_position_embeddings is 16384; CUAD at 33 K would OOM the
    model config.
    """
    assert _max_model_len_for("unsloth/phi-4-unsloth-bnb-4bit", default=131072) == 16384


def test_max_model_len_does_not_cap_llama_3_3_70b():
    """llama-3.3-70b is NOT in the cap table : with tensor_parallel_size=3
    on three 48 GB cards the KV cache budget covers CUAD's 33 K context.
    Verifying the absence of a cap so a future "let's be safe and add
    11 K back" mutation gets caught.
    """
    assert (
        _max_model_len_for("casperhansen/llama-3.3-70b-instruct-awq", default=33280)
        == 33280
    )


def test_max_model_len_cap_overrides_higher_default():
    """A caller asking for 32 K on phi-4 still gets capped at 16 K
    (the cap is a ceiling, not a default).
    """
    assert _max_model_len_for("stelterlab/phi-4-AWQ", default=32768) == 16384


def test_max_model_len_unknown_model_returns_default():
    """No cap match -> caller's default wins."""
    assert _max_model_len_for("unknown/some-model", default=12345) == 12345


# ----------------------------------------------------------------------
# _quantization_for : pick the right vLLM quantization arg from model_id
# ----------------------------------------------------------------------


def test_quantization_for_bnb_default():
    """Default path: Unsloth Dynamic 2.0 bnb checkpoints -> bitsandbytes."""
    assert _quantization_for("unsloth/Qwen3-14B-unsloth-bnb-4bit") == "bitsandbytes"
    assert (
        _quantization_for("unsloth/granite-3.2-8b-instruct-unsloth-bnb-4bit")
        == "bitsandbytes"
    )


def test_quantization_for_awq_suffix():
    """AWQ checkpoints must use awq_marlin so vLLM avoids the broken bnb_loader.

    Concrete case: casperhansen/llama-3.3-70b-instruct-awq -- the bnb_loader
    has a known shape mismatch on Llama-3.3 70B GQA qkv_proj, so we have to
    route these through the AWQ kernels instead.
    """
    assert _quantization_for("casperhansen/llama-3.3-70b-instruct-awq") == "awq_marlin"
    assert (
        _quantization_for("casperhansen/deepseek-r1-distill-llama-70b-awq")
        == "awq_marlin"
    )


def test_quantization_for_awq_int4_variant():
    """Some publishers use the -awq-int4 suffix."""
    assert _quantization_for("some/Model-AWQ-int4") == "awq_marlin"


def test_quantization_for_awq_in_middle_of_name():
    """The substring '-awq-' anywhere in the id should still route to AWQ."""
    assert _quantization_for("foo/bar-awq-quantized") == "awq_marlin"


def test_quantization_for_gptq_suffix():
    """GPTQ uses gptq_marlin (faster than the legacy gptq kernel)."""
    assert _quantization_for("TheBloke/Llama-2-7B-GPTQ") == "gptq_marlin"


def test_quantization_for_is_case_insensitive():
    """Suffix detection must not depend on the publisher's casing."""
    assert _quantization_for("Some/Llama-AWQ") == "awq_marlin"
    assert _quantization_for("Some/Llama-GPTQ") == "gptq_marlin"


def test_quantization_for_gpt_oss_returns_none():
    """gpt-oss-* ships MXFP4 natively; the default branch returns None so
    vLLM auto-detects the quantization from config.json. Crashed under
    the old 'bitsandbytes' default before PR #35.
    """
    assert _quantization_for("openai/gpt-oss-20b") is None
    assert _quantization_for("openai/gpt-oss-120b") is None
    assert _quantization_for("openai/GPT-OSS-20B") is None  # case insensitive


def test_quantization_for_fp16_ibm_granite_returns_none():
    """ibm-granite/granite-3.2-8b-instruct (FP16, no quant suffix) must
    default to None so vLLM auto-detects -- not bitsandbytes, which
    would crash the engine.
    """
    assert _quantization_for("ibm-granite/granite-3.2-8b-instruct") is None


def test_quantization_for_fp8_dynamic():
    """RedHatAI/...FP8-dynamic checkpoints route through fp8 kernels."""
    assert _quantization_for("RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic") == "fp8"
    assert _quantization_for("RedHatAI/QwQ-32B-FP8-dynamic") == "fp8"


def test_quantization_for_fp8():
    """FP8 checkpoints (e.g. Qwen/Qwen3-30B-A3B-FP8) need quantization='fp8'
    so vLLM routes through the fp8 kernels (TP-compatible on Ada+).
    """
    assert _quantization_for("Qwen/Qwen3-30B-A3B-FP8") == "fp8"
    assert _quantization_for("some-model-fp8") == "fp8"
    assert _quantization_for("some-fp8-variant") == "fp8"


def test_quantization_for_w4a16_compressed_tensors():
    """RedHatAI ...w4a16 checkpoints route through compressed-tensors so
    vLLM uses the marlin kernels that support TP. bitsandbytes does not.
    """
    assert (
        _quantization_for(
            "RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16"
        )
        == "compressed-tensors"
    )


def test_load_local_vllm_caps_tp_to_1_for_bitsandbytes():
    """vLLM refuses TP>1 on bitsandbytes checkpoints. The wrapper must
    silently drop to TP=1 so an aggressive --tensor-parallel-size on
    the CLI does not crash granite-3.2-8b (the only bnb-only model
    left after PR #34 swaps gemma-2 -> gemma-3 GPTQ and qwen3-30b-a3b
    -> FP8)."""
    import types
    from unittest.mock import patch

    vllm_mod._vllm_engine.clear()

    captured: dict = {}

    def fake_llm(**kw):
        captured.update(kw)
        return object()

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = fake_llm

    with patch.dict("sys.modules", {"vllm": fake_vllm}):
        from experiments.evaluate_local_vllm import _load_local_vllm

        _load_local_vllm(
            "unsloth/granite-3.2-8b-instruct-unsloth-bnb-4bit",
            tensor_parallel_size=3,
        )

    assert captured["tensor_parallel_size"] == 1, (
        "bnb checkpoint must be auto-capped to TP=1 to avoid "
        "'Prequant BitsAndBytes models with tensor parallelism is "
        "not supported' from vLLM"
    )


def test_load_local_vllm_keeps_tp_for_awq():
    """AWQ checkpoints support TP>1 -- the cap must NOT trigger."""
    import types
    from unittest.mock import patch

    vllm_mod._vllm_engine.clear()

    captured: dict = {}

    def fake_llm(**kw):
        captured.update(kw)
        return object()

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = fake_llm

    with patch.dict("sys.modules", {"vllm": fake_vllm}):
        from experiments.evaluate_local_vllm import _load_local_vllm

        _load_local_vllm(
            "casperhansen/llama-3.3-70b-instruct-awq",
            tensor_parallel_size=2,
        )

    assert captured["tensor_parallel_size"] == 2
    assert captured["quantization"] == "awq_marlin"


# ----------------------------------------------------------------------
# _load_local_vllm : tensor_parallel_size propagation
# ----------------------------------------------------------------------


def test_load_local_vllm_rejects_tp_zero():
    """tensor_parallel_size must be >= 1."""
    import pytest

    from experiments.evaluate_local_vllm import _load_local_vllm

    with pytest.raises(ValueError, match="tensor_parallel_size"):
        _load_local_vllm("unsloth/whatever", tensor_parallel_size=0)


def test_load_local_vllm_rejects_tp_negative():
    """Negative tensor_parallel_size is nonsense."""
    import pytest

    from experiments.evaluate_local_vllm import _load_local_vllm

    with pytest.raises(ValueError, match="tensor_parallel_size"):
        _load_local_vllm("unsloth/whatever", tensor_parallel_size=-2)


def test_load_local_vllm_reads_tp_from_env():
    """SEVERITY_EVAL_TP env var sets tensor_parallel_size when caller
    leaves it at the default (None).

    This is how evaluate_models.py --tensor-parallel-size threads the
    setting down to the LLM constructor without changing the public
    evaluate_local_vllm signature.
    """
    import os
    import types
    from unittest.mock import patch

    vllm_mod._vllm_engine.clear()

    captured: dict = {}

    def fake_llm(**kw):
        captured.update(kw)
        return object()

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = fake_llm

    with (
        patch.dict("sys.modules", {"vllm": fake_vllm}),
        patch.dict(os.environ, {"SEVERITY_EVAL_TP": "3"}, clear=False),
    ):
        from experiments.evaluate_local_vllm import _load_local_vllm

        _load_local_vllm("foo/test-model-awq")

    assert captured["tensor_parallel_size"] == 3, (
        "SEVERITY_EVAL_TP=3 should land in the LLM(...) call as "
        f"tensor_parallel_size=3, got {captured.get('tensor_parallel_size')!r}"
    )


def test_load_local_vllm_cache_invalidates_when_tp_differs():
    """Cache must be invalidated when tensor_parallel_size differs from
    the cached engine's TP.

    Concrete scenario: phase 1 loads a model with TP=1, then a phase 2
    script asks for the same model with TP=3. Without checking the TP
    field the cache would hand back the TP=1 engine and the TP=3 run
    would silently use the wrong engine.

    Use an AWQ-suffixed model_id so the bnb-only TP=1 cap does not
    fire and TP=3 actually reaches the LLM constructor.
    """
    import types
    from unittest.mock import patch

    # Pre-populate cache with a TP=1 engine
    vllm_mod._vllm_engine.clear()
    stale_llm = object()
    vllm_mod._vllm_engine["model_id"] = "casperhansen/test-model-awq"
    vllm_mod._vllm_engine["max_model_len"] = 8192
    vllm_mod._vllm_engine["tensor_parallel_size"] = 1
    vllm_mod._vllm_engine["llm"] = stale_llm

    new_llm = object()
    captured: dict = {}

    def fake_llm(**kw):
        captured.update(kw)
        return new_llm

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = fake_llm

    with patch.dict("sys.modules", {"vllm": fake_vllm}):
        from experiments.evaluate_local_vllm import _load_local_vllm

        result = _load_local_vllm("casperhansen/test-model-awq", tensor_parallel_size=3)

    assert result is not stale_llm, "cache must invalidate when TP changes"
    assert result is new_llm
    assert captured["tensor_parallel_size"] == 3
    assert vllm_mod._vllm_engine["tensor_parallel_size"] == 3


def test_load_local_vllm_cache_hits_when_tp_matches():
    """Same model_id, max_model_len fits, and TP matches -> cache hit.

    Use AWQ so the bnb TP=1 cap does not interfere.
    """
    vllm_mod._vllm_engine.clear()
    cached_llm = object()
    vllm_mod._vllm_engine["model_id"] = "casperhansen/test-model-awq"
    vllm_mod._vllm_engine["max_model_len"] = 32768
    vllm_mod._vllm_engine["tensor_parallel_size"] = 2
    vllm_mod._vllm_engine["llm"] = cached_llm

    from experiments.evaluate_local_vllm import _load_local_vllm

    # Smaller max_len + same TP -> serve cached
    result = _load_local_vllm(
        "casperhansen/test-model-awq", max_model_len=4096, tensor_parallel_size=2
    )
    assert result is cached_llm


def test_load_local_vllm_explicit_tp_wins_over_env():
    """Explicit tensor_parallel_size param takes precedence over the env var."""
    import os
    import types
    from unittest.mock import patch

    vllm_mod._vllm_engine.clear()

    captured: dict = {}

    def fake_llm(**kw):
        captured.update(kw)
        return object()

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = fake_llm

    with (
        patch.dict("sys.modules", {"vllm": fake_vllm}),
        patch.dict(os.environ, {"SEVERITY_EVAL_TP": "3"}, clear=False),
    ):
        from experiments.evaluate_local_vllm import _load_local_vllm

        _load_local_vllm("foo/test-model-awq", tensor_parallel_size=2)

    assert captured["tensor_parallel_size"] == 2


# ----------------------------------------------------------------------
# _load_local_vllm — argument validation + cache invalidation
# ----------------------------------------------------------------------


def test_load_local_vllm_rejects_gpu_util_above_one():
    """gpu_memory_utilization > 1.0 must raise ValueError before touching vLLM."""
    import pytest

    from experiments.evaluate_local_vllm import _load_local_vllm

    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        _load_local_vllm("unsloth/whatever", gpu_memory_utilization=1.5)


def test_load_local_vllm_rejects_gpu_util_zero():
    """gpu_memory_utilization == 0 is meaningless (no memory available)."""
    import pytest

    from experiments.evaluate_local_vllm import _load_local_vllm

    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        _load_local_vllm("unsloth/whatever", gpu_memory_utilization=0.0)


def test_load_local_vllm_rejects_gpu_util_negative():
    """Negative gpu_memory_utilization is nonsense."""
    import pytest

    from experiments.evaluate_local_vllm import _load_local_vllm

    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        _load_local_vllm("unsloth/whatever", gpu_memory_utilization=-0.1)


def test_load_local_vllm_cache_invalidates_when_max_len_grows():
    """If a second call requests a larger max_model_len, the cache must be
    invalidated so the engine is reloaded with the bigger context.

    Concrete scenario: smoke runs MedQA (4 K) then CUAD (33 K) on the same
    model. Reusing the 4 K engine would crash on a long CUAD prompt.
    """
    import types
    from unittest.mock import patch

    # Pre-populate cache with a "small-context" sentinel engine
    vllm_mod._vllm_engine.clear()
    sentinel_llm = object()
    vllm_mod._vllm_engine["model_id"] = "unsloth/test-model"
    vllm_mod._vllm_engine["max_model_len"] = 4096
    vllm_mod._vllm_engine["llm"] = sentinel_llm

    # Stub vllm.LLM to return a sentinel without touching CUDA. A
    # types.ModuleType is necessary because a class-attribute lambda
    # gets the method-binding treatment (self is passed); module
    # attributes do not.
    new_llm = object()
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = lambda **kw: new_llm

    with patch.dict("sys.modules", {"vllm": fake_vllm}):
        from experiments.evaluate_local_vllm import _load_local_vllm

        result = _load_local_vllm("unsloth/test-model", max_model_len=32768)
    assert result is not sentinel_llm, "stale 4K engine reused for 32K request"
    assert result is new_llm
    assert vllm_mod._vllm_engine["max_model_len"] == 32768


def test_load_local_vllm_falls_back_on_runtime_error():
    """vLLM raises a RuntimeError ("Engine core initialization failed") in
    the parent when the EngineCore sub-process dies on an AssertionError
    inside the bnb weight loader (Unsloth Dynamic 2.0 shape mismatch).

    Concrete case: unsloth/Llama-3.3-70B-Instruct-unsloth-bnb-4bit and
    unsloth/DeepSeek-R1-Distill-Llama-70B-unsloth-bnb-4bit. The fallback
    must try the non-Dynamic -bnb-4bit repo before giving up.
    """
    import types
    from unittest.mock import patch

    vllm_mod._vllm_engine.clear()

    fallback_llm = object()
    call_log: list[str] = []

    def fake_llm(**kw):
        repo = kw["model"]
        call_log.append(repo)
        if repo.endswith("-unsloth-bnb-4bit"):
            raise RuntimeError(
                "Engine core initialization failed. See root cause above."
            )
        return fallback_llm

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = fake_llm

    with patch.dict("sys.modules", {"vllm": fake_vllm}):
        from experiments.evaluate_local_vllm import _load_local_vllm

        llm = _load_local_vllm("unsloth/Llama-3.3-70B-Instruct-unsloth-bnb-4bit")

    assert llm is fallback_llm
    assert call_log == [
        "unsloth/Llama-3.3-70B-Instruct-unsloth-bnb-4bit",
        "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
    ]
    # Cache must record the fallback repo, not the Dynamic one we rejected
    assert (
        vllm_mod._vllm_engine["model_id"] == "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
    )


def test_load_local_vllm_falls_back_on_assertion_error():
    """Direct AssertionError from a synchronous call path also triggers
    the fallback. Some vLLM versions raise it in-process when the
    EngineCore is run inline (uniproc executor without subprocess)."""
    import types
    from unittest.mock import patch

    vllm_mod._vllm_engine.clear()

    fallback_llm = object()
    call_log: list[str] = []

    def fake_llm(**kw):
        repo = kw["model"]
        call_log.append(repo)
        if repo.endswith("-unsloth-bnb-4bit"):
            raise AssertionError()
        return fallback_llm

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = fake_llm

    with patch.dict("sys.modules", {"vllm": fake_vllm}):
        from experiments.evaluate_local_vllm import _load_local_vllm

        llm = _load_local_vllm("unsloth/DeepSeek-R1-Distill-Llama-70B-unsloth-bnb-4bit")

    assert llm is fallback_llm
    assert call_log[-1] == "unsloth/DeepSeek-R1-Distill-Llama-70B-bnb-4bit"


def test_load_local_vllm_runtime_error_propagates_if_no_unsloth_suffix():
    """If the repo does not have an -unsloth-bnb-4bit suffix, a
    RuntimeError must propagate -- there is no fallback to try."""
    import types
    from unittest.mock import patch

    import pytest

    vllm_mod._vllm_engine.clear()

    def fake_llm(**kw):
        raise RuntimeError("genuine engine failure")

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = fake_llm

    with patch.dict("sys.modules", {"vllm": fake_vllm}):
        from experiments.evaluate_local_vllm import _load_local_vllm

        with pytest.raises(RuntimeError, match="genuine engine failure"):
            _load_local_vllm("meta-llama/Llama-3.3-70B")


def test_load_local_vllm_cache_hits_when_max_len_fits():
    """Cache hit: same model_id and the cached engine already supports the
    requested context (or more) -- return the existing engine."""
    vllm_mod._vllm_engine.clear()
    sentinel_llm = object()
    vllm_mod._vllm_engine["model_id"] = "unsloth/test-model"
    vllm_mod._vllm_engine["max_model_len"] = 32768
    vllm_mod._vllm_engine["llm"] = sentinel_llm

    from experiments.evaluate_local_vllm import _load_local_vllm

    # Asking for a smaller context -- cache should serve the existing engine
    result = _load_local_vllm("unsloth/test-model", max_model_len=4096)
    assert result is sentinel_llm
    # And the cached state is untouched
    assert vllm_mod._vllm_engine["max_model_len"] == 32768


# ----------------------------------------------------------------------
# _destroy_engine — idempotent + clears the cache
# ----------------------------------------------------------------------


def test_destroy_engine_is_noop_when_no_engine_cached():
    """No engine = early return without touching anything."""
    # Ensure cache empty
    vllm_mod._vllm_engine.clear()
    # Should not raise even if torch / vllm are absent
    _destroy_engine()
    assert vllm_mod._vllm_engine == {}


def test_destroy_engine_clears_cached_entries():
    """When an engine is cached, destroy empties the dict."""
    vllm_mod._vllm_engine["model_id"] = "dummy"
    vllm_mod._vllm_engine["llm"] = object()
    # Patch the optional imports to avoid needing the actual libs
    with patch("builtins.__import__") as mock_import:
        mock_import.side_effect = ImportError("not installed")
        _destroy_engine()
    assert vllm_mod._vllm_engine == {}


def test_destroy_engine_calls_distributed_cleanup_when_available():
    """If vllm.distributed is importable, destroy_distributed_environment runs."""
    vllm_mod._vllm_engine["model_id"] = "dummy"
    vllm_mod._vllm_engine["llm"] = object()
    called = {"count": 0}

    def fake_destroy():
        called["count"] += 1

    real_import = __import__

    def selective_import(name, *args, **kwargs):
        if name == "vllm.distributed":

            class _M:
                destroy_distributed_environment = staticmethod(fake_destroy)

            return _M
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=selective_import):
        _destroy_engine()

    assert called["count"] == 1
    assert vllm_mod._vllm_engine == {}


def test_destroy_engine_swallows_runtime_error_from_cleanup():
    """If destroy_distributed_environment raises RuntimeError, we still
    clear the cache (the engine is being torn down anyway)."""
    vllm_mod._vllm_engine["model_id"] = "dummy"
    vllm_mod._vllm_engine["llm"] = object()
    real_import = __import__

    def selective_import(name, *args, **kwargs):
        if name == "vllm.distributed":

            class _M:
                @staticmethod
                def destroy_distributed_environment():
                    raise RuntimeError("torn down")

            return _M
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=selective_import):
        _destroy_engine()  # must not propagate the RuntimeError

    assert vllm_mod._vllm_engine == {}
