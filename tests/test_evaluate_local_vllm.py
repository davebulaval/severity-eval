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
    default. granite-3.2-8b is not in the cap table.
    """
    assert (
        _max_model_len_for("ibm-granite/granite-3.2-8b-instruct", default=33280)
        == 33280
    )


def test_max_model_len_caps_qwen25_72b_at_32k():
    """Qwen2.5-72B-Instruct has max_position_embeddings=32768. CUAD asks
    for ~33 K, so vLLM rejects unless we pass max_model_len <= 32768.
    """
    assert (
        _max_model_len_for("RedHatAI/Qwen2.5-72B-Instruct-FP8-dynamic", default=131072)
        == 32768
    )


def test_max_model_len_caps_qwen3_at_32k():
    """Qwen3 family (Qwen3-14B, Qwen3-30B-A3B) tops out at 32 K natively."""
    assert _max_model_len_for("Qwen/Qwen3-14B-FP8", default=131072) == 32768
    assert _max_model_len_for("Qwen/Qwen3-30B-A3B-FP8", default=131072) == 32768


def test_max_model_len_caps_qwq_at_32k():
    """QwQ-32B inherits Qwen2's 32 K cap."""
    assert _max_model_len_for("RedHatAI/QwQ-32B-FP8-dynamic", default=131072) == 32768


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


def test_quantization_for_fp8_returns_none_for_autodetect():
    """FP8 packaging varies by publisher (Qwen ships native fp8, RedHat
    ships compressed-tensors with fp8 weights). We return None so vLLM
    reads quant_method from config.json -- otherwise vLLM raises
    'Quantization method specified in the model config (compressed-tensors)
    does not match the quantization method specified in the quantization
    argument (fp8)' for RedHat FP8-dynamic.
    """
    assert _quantization_for("RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic") is None
    assert _quantization_for("RedHatAI/QwQ-32B-FP8-dynamic") is None
    assert _quantization_for("Qwen/Qwen3-30B-A3B-FP8") is None
    assert _quantization_for("some-model-fp8") is None
    assert _quantization_for("some-fp8-variant") is None


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


# ----------------------------------------------------------------------
# evaluate_local_vllm : checkpoint + resume + extend
# ----------------------------------------------------------------------


class _FakeOutput:
    """Mimics vllm.RequestOutput just enough for evaluate_local_vllm."""

    class _Choice:
        def __init__(self, text: str):
            self.text = text

    def __init__(self, text: str):
        self.outputs = [self._Choice(text)]


class _FakeLLM:
    """Stand-in for vllm.LLM that records every generate() call."""

    def __init__(self, return_text: str = "ANSWER"):
        self.calls: list[dict] = []
        self.return_text = return_text

    def get_tokenizer(self):
        class _Tok:
            @staticmethod
            def apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ):
                return messages[0]["content"]

            @staticmethod
            def encode(text, add_special_tokens=False):
                # 1 token per character - cheap, deterministic, well below
                # any model's max_model_len for the test prompts we use.
                return list(range(min(len(text), 20)))

        return _Tok()

    def generate(self, *args, **kw):
        self.calls.append({"args": args, "kw": kw})
        # Figure out batch size from args or kw
        if args:
            payload = args[0]
        elif "prompt_token_ids" in kw:
            payload = kw["prompt_token_ids"]
        else:
            payload = []
        return [_FakeOutput(self.return_text) for _ in payload]


def _make_df(n: int, dataset: str = "medqa"):
    """Make a tiny DataFrame that evaluate_local_vllm accepts."""
    import pandas as pd

    rows = []
    for i in range(n):
        rows.append(
            {
                "id": f"q{i}",
                "question": f"What is {i}+1?",
                "answer": str(i + 1),
                "severity": "minor",
                "domain": "math",
            }
        )
    return pd.DataFrame(rows)


def _patch_eval_deps(monkeypatch, llm: _FakeLLM):
    """Stub the heavy dependencies of evaluate_local_vllm so the function
    runs end-to-end in a unit test without vLLM, transformers, or CUDA."""
    import sys
    import types

    # Stub vllm.SamplingParams
    fake_vllm = sys.modules.get("vllm") or types.ModuleType("vllm")
    fake_vllm.SamplingParams = lambda **kw: ("SAMPLING_PARAMS", kw)
    sys.modules["vllm"] = fake_vllm

    # Stub the helpers imported lazily from evaluate_models.
    from experiments import evaluate_models as em_real

    monkeypatch.setattr(
        em_real, "_is_thinking_model", lambda model_id: False, raising=False
    )
    monkeypatch.setattr(
        em_real,
        "_build_prompt_for_row",
        lambda row, ds, style: (row["question"], None),
        raising=False,
    )
    monkeypatch.setattr(
        em_real,
        "_strip_think_tags",
        lambda text: text,
        raising=False,
    )
    monkeypatch.setattr(
        em_real,
        "score_prediction",
        lambda pred, ans, options=None: {
            "correct": pred.strip() == ans.strip(),
            "score_method": "exact_match",
        },
        raising=False,
    )
    monkeypatch.setattr(
        em_real,
        "DATASET_INFERENCE_CONFIG",
        {"medqa": (16, 1024)},
        raising=False,
    )
    monkeypatch.setattr(
        em_real,
        "_DEFAULT_INFERENCE_CONFIG",
        (16, 1024),
        raising=False,
    )
    monkeypatch.setattr(em_real, "_THINKING_TOKEN_MULTIPLIER", 16, raising=False)
    monkeypatch.setattr(em_real, "_THINKING_MAX_NEW_TOKENS_CAP", 32000, raising=False)

    # Replace _load_local_vllm so we don't touch CUDA
    monkeypatch.setattr(vllm_mod, "_load_local_vllm", lambda *a, **kw: llm)


def test_evaluate_local_vllm_writes_checkpoint_after_each_chunk(tmp_path, monkeypatch):
    """A 25-row dataset with chunk_size=10 must write the JSON 3 times
    (one per chunk : 10, 20, 25 rows)."""
    from experiments.evaluate_local_vllm import evaluate_local_vllm

    llm = _FakeLLM(return_text="x")
    _patch_eval_deps(monkeypatch, llm)

    out = tmp_path / "medqa_test.json"
    df = _make_df(25)

    # Record every write to the output file
    import json

    n_rows_per_write: list[int] = []

    real_write_text = type(out).write_text

    def spy_write_text(self, content, *a, **kw):
        if str(self).endswith(".tmp"):
            try:
                n_rows_per_write.append(len(json.loads(content)))
            except Exception:
                pass
        return real_write_text(self, content, *a, **kw)

    monkeypatch.setattr(type(out), "write_text", spy_write_text)

    results = evaluate_local_vllm(
        df=df,
        model_name="test-model",
        model_id="foo/bar",
        dataset_name="medqa",
        prompt_style="original",
        output_path=out,
        chunk_size=10,
    )

    assert len(results) == 25
    # 3 chunks -> 3 checkpoint saves
    assert llm.calls and len(llm.calls) == 3, f"expected 3 chunks, got {len(llm.calls)}"
    assert n_rows_per_write == [10, 20, 25], (
        f"checkpoints should grow 10 -> 20 -> 25 rows, got {n_rows_per_write}"
    )
    # Final file present and complete
    final = json.loads(out.read_text())
    assert len(final) == 25
    assert {r["id"] for r in final} == {f"q{i}" for i in range(25)}


def test_evaluate_local_vllm_resumes_from_existing_output(tmp_path, monkeypatch):
    """When the output_path already has 5 entries and df has 10 rows,
    only the 5 new ids are sent to vLLM."""
    import json

    from experiments.evaluate_local_vllm import evaluate_local_vllm

    llm = _FakeLLM(return_text="x")
    _patch_eval_deps(monkeypatch, llm)

    out = tmp_path / "medqa_test.json"
    # Seed an existing partial result with the first 5 rows already done
    seeded = [
        {
            "id": f"q{i}",
            "question": f"What is {i}+1?",
            "answer": str(i + 1),
            "severity": "minor",
            "domain": "math",
            "model": "test-model",
            "prediction": "SEEDED",
            "correct": False,
            "score_method": "exact_match",
        }
        for i in range(5)
    ]
    out.write_text(json.dumps(seeded))

    results = evaluate_local_vllm(
        df=_make_df(10),
        model_name="test-model",
        model_id="foo/bar",
        dataset_name="medqa",
        prompt_style="original",
        output_path=out,
        chunk_size=100,
    )

    assert len(results) == 10
    # Only 5 new prompts should have been submitted to vLLM
    submitted = sum(
        len(c["args"][0] if c["args"] else c["kw"].get("prompt_token_ids", []))
        for c in llm.calls
    )
    assert submitted == 5, f"expected 5 new prompts, got {submitted}"

    # The seeded predictions must be preserved verbatim (not overwritten)
    by_id = {r["id"]: r for r in results}
    assert by_id["q0"]["prediction"] == "SEEDED"
    assert by_id["q4"]["prediction"] == "SEEDED"
    # And new rows have the fake LLM output
    assert by_id["q5"]["prediction"] == "x"
    assert by_id["q9"]["prediction"] == "x"


def test_evaluate_local_vllm_skips_when_all_done(tmp_path, monkeypatch):
    """If every df id is already in output_path, vLLM must not be called."""
    import json

    from experiments.evaluate_local_vllm import evaluate_local_vllm

    llm = _FakeLLM(return_text="x")
    _patch_eval_deps(monkeypatch, llm)

    out = tmp_path / "medqa_test.json"
    seeded = [
        {
            "id": f"q{i}",
            "question": f"q{i}",
            "answer": str(i + 1),
            "severity": "minor",
            "domain": "math",
            "model": "test-model",
            "prediction": "DONE",
            "correct": True,
            "score_method": "exact_match",
        }
        for i in range(5)
    ]
    out.write_text(json.dumps(seeded))

    results = evaluate_local_vllm(
        df=_make_df(5),
        model_name="test-model",
        model_id="foo/bar",
        dataset_name="medqa",
        prompt_style="original",
        output_path=out,
    )

    assert len(results) == 5
    assert llm.calls == [], "no vLLM call should be made when all ids done"


def test_evaluate_local_vllm_force_ignores_existing(tmp_path, monkeypatch):
    """force=True must re-run every row even if output_path is fully populated."""
    import json

    from experiments.evaluate_local_vllm import evaluate_local_vllm

    llm = _FakeLLM(return_text="FRESH")
    _patch_eval_deps(monkeypatch, llm)

    out = tmp_path / "medqa_test.json"
    out.write_text(
        json.dumps(
            [
                {
                    "id": f"q{i}",
                    "question": "x",
                    "answer": "y",
                    "severity": "minor",
                    "domain": "math",
                    "model": "m",
                    "prediction": "STALE",
                    "correct": False,
                    "score_method": "exact_match",
                }
                for i in range(5)
            ]
        )
    )

    results = evaluate_local_vllm(
        df=_make_df(5),
        model_name="test-model",
        model_id="foo/bar",
        dataset_name="medqa",
        prompt_style="original",
        output_path=out,
        force=True,
    )

    assert len(results) == 5
    assert all(r["prediction"] == "FRESH" for r in results), (
        "force=True should overwrite STALE seeded predictions"
    )


def test_atomic_write_json_uses_tmp_then_replace(tmp_path):
    """The helper must write through a .tmp suffix so a half-written file
    never replaces a good one on crash."""
    from experiments.evaluate_local_vllm import _atomic_write_json

    target = tmp_path / "out.json"
    _atomic_write_json(target, [{"id": 1, "v": "a"}, {"id": 2, "v": "b"}])
    assert target.exists()
    import json

    assert json.loads(target.read_text()) == [{"id": 1, "v": "a"}, {"id": 2, "v": "b"}]
    # No leftover tmp file
    assert not (tmp_path / "out.json.tmp").exists()


def test_evaluate_local_vllm_handles_corrupt_existing(tmp_path, monkeypatch):
    """A garbage output_path should be treated as 'start fresh', not crash."""
    from experiments.evaluate_local_vllm import evaluate_local_vllm

    llm = _FakeLLM(return_text="x")
    _patch_eval_deps(monkeypatch, llm)

    out = tmp_path / "medqa_test.json"
    out.write_text("not valid json {{{")

    results = evaluate_local_vllm(
        df=_make_df(3),
        model_name="test-model",
        model_id="foo/bar",
        dataset_name="medqa",
        prompt_style="original",
        output_path=out,
        chunk_size=10,
    )
    assert len(results) == 3
    # 1 chunk of 3 prompts
    assert len(llm.calls) == 1


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
