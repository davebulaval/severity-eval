"""Tests for the pure-logic helpers in experiments.evaluate_local_vllm.

The actual engine loading and inference are not unit tested because they
require vLLM + a GPU. We cover:
    - _max_model_len_for: model-specific context caps
    - _destroy_engine: idempotent cleanup of the global cache
"""

from __future__ import annotations

from unittest.mock import patch

import experiments.evaluate_local_vllm as vllm_mod
from experiments.evaluate_local_vllm import _destroy_engine, _max_model_len_for


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
