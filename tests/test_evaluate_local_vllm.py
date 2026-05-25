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


def test_max_model_len_caps_gemma_2_9b_below_default():
    """gemma-2-9b must be capped at 8K even when caller asks for more.

    Pass an explicit default greater than 8K so this test fails if the
    'if "gemma-2" in model_id' branch is removed (mutation mental check).
    """
    assert (
        _max_model_len_for("unsloth/gemma-2-9b-it-unsloth-bnb-4bit", default=32768)
        == 8192
    )


def test_max_model_len_caps_gemma_2_27b_below_default():
    """gemma-2-27b also matches the gemma-2 pattern and is capped at 8K."""
    assert (
        _max_model_len_for("unsloth/gemma-2-27b-it-unsloth-bnb-4bit", default=32768)
        == 8192
    )


def test_max_model_len_gemma_cap_overrides_higher_default():
    """Even if the caller asks for 32k, gemma-2 stays at 8k.

    Tests an invariant: gemma-2 must not be allowed to exceed its native
    8K RoPE window, or generation produces garbled output.
    """
    assert _max_model_len_for("gemma-2-anything", default=131072) == 8192


def test_max_model_len_non_gemma_uses_default():
    assert _max_model_len_for("unsloth/Qwen3-14B-unsloth-bnb-4bit") == 8192


def test_max_model_len_non_gemma_respects_caller_default():
    assert (
        _max_model_len_for("unsloth/Llama-3.3-70B-Instruct-bnb-4bit", default=32768)
        == 32768
    )


def test_max_model_len_case_insensitive_for_gemma():
    """Gemma-2 detection is case-insensitive."""
    assert _max_model_len_for("Unsloth/GEMMA-2-9b-It", default=131072) == 8192


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
