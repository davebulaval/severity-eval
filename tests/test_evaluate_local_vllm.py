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
