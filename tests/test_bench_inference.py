"""Tests for experiments.bench_inference.

The benchmark dispatcher itself loads a real model, so it is not unit
tested. We exercise the pure-logic helpers:
    - _git: returns "unknown" on git failure, captures real values otherwise
    - _model_id_for: looks up a model name and rejects non-local providers
    - _save: writes a JSON file with the expected name shape
    - compare: produces the right FASTER/SLOWER annotations
    - run_benchmark: rejects n_samples < 1 before touching CUDA
    - PROMPTS: present and well-formed
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest

from experiments.bench_inference import (
    PROMPTS,
    _git,
    _model_id_for,
    _save,
    compare,
    run_benchmark,
)


# ----------------------------------------------------------------------
# PROMPTS — fixed list must be non-empty and contain only non-empty strings
# ----------------------------------------------------------------------


def test_prompts_are_non_empty():
    assert len(PROMPTS) >= 4
    assert all(isinstance(p, str) for p in PROMPTS)
    assert all(len(p.strip()) > 0 for p in PROMPTS)


def test_prompts_cover_paper_task_types():
    """Heuristic: prompts should reference key paper concepts at least once."""
    blob = " ".join(PROMPTS).lower()
    # At least one MCQ-style prompt
    assert any(
        "a)" in p.lower() or "answer with the letter" in p.lower() for p in PROMPTS
    )
    # At least one numeric extraction
    assert "$" in blob or "percentage" in blob
    # At least one yes/no question
    assert "yes or no" in blob or "yes/no" in blob


# ----------------------------------------------------------------------
# _git — returns "unknown" on failure, real commit otherwise
# ----------------------------------------------------------------------


def test_git_returns_unknown_when_subprocess_fails():
    with patch("experiments.bench_inference.subprocess.check_output") as mock_run:
        mock_run.side_effect = FileNotFoundError("git not installed")
        assert _git("rev-parse", "HEAD") == "unknown"


def test_git_returns_unknown_on_called_process_error():
    import subprocess

    with patch("experiments.bench_inference.subprocess.check_output") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, "git")
        assert _git("status") == "unknown"


def test_git_returns_stripped_stdout_on_success():
    with patch("experiments.bench_inference.subprocess.check_output") as mock_run:
        mock_run.return_value = b"abcd1234\n"
        assert _git("rev-parse", "HEAD") == "abcd1234"


# ----------------------------------------------------------------------
# _model_id_for — resolves logical model_name to HF repo id
# ----------------------------------------------------------------------


def test_model_id_for_returns_hf_id_for_local_model():
    # 'gemma-3-12b' is a known local model in MODELS (gemma-3 replaced
    # the gemma-2 entries). Resolver should return something gemma-shaped.
    result = _model_id_for("gemma-3-12b")
    assert "gemma" in result.lower()
    # Must point at a non-empty HF org/repo with a slash
    assert "/" in result


def test_model_id_for_rejects_unknown_model():
    with pytest.raises(SystemExit, match="Unknown model"):
        _model_id_for("not-a-real-model-name")


def test_model_id_for_rejects_api_provider_model():
    # 'gpt-5' is API (openai), not local
    with pytest.raises(SystemExit, match="bench_inference targets only local"):
        _model_id_for("gpt-5")


# ----------------------------------------------------------------------
# _save — writes JSON with expected filename shape
# ----------------------------------------------------------------------


def test_save_writes_file_with_branch_commit_model_in_name(tmp_path, monkeypatch):
    # Redirect BENCH_DIR to tmp
    monkeypatch.setattr("experiments.bench_inference.BENCH_DIR", tmp_path)
    metrics = {
        "git_branch": "speedup/test",
        "git_commit": "abc12345",
        "model_name": "qwen3-14b",
        "timestamp_utc": "2026-05-24T10:30:00+00:00",
        "load_seconds": 12.3,
    }
    path = _save(metrics)
    assert path.exists()
    # branch slash is replaced
    assert "speedup_test" in path.name
    assert "abc12345" in path.name
    assert "qwen3-14b" in path.name
    # timestamp colons replaced for filename safety
    assert ":" not in path.name
    # content roundtrips
    assert json.loads(path.read_text())["load_seconds"] == 12.3


def test_save_creates_bench_dir_if_missing(tmp_path, monkeypatch):
    target = tmp_path / "nested" / "dir"
    monkeypatch.setattr("experiments.bench_inference.BENCH_DIR", target)
    metrics = {
        "git_branch": "main",
        "git_commit": "deadbeef",
        "model_name": "phi-4",
        "timestamp_utc": "2026-05-24T10:30:00+00:00",
    }
    _save(metrics)
    assert target.is_dir()


# ----------------------------------------------------------------------
# compare — prints FASTER / SLOWER per metric
# ----------------------------------------------------------------------


def _write_bench(tmp_path: Path, name: str, payload: dict) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(payload))
    return p


@pytest.fixture
def baseline_json(tmp_path):
    return _write_bench(
        tmp_path,
        "baseline.json",
        {
            "model_name": "qwen3-14b",
            "load_seconds": 60.0,
            "total_seconds": 800.0,
            "mean_latency_seconds": 100.0,
            "p50_latency_seconds": 90.0,
            "p90_latency_seconds": 140.0,
            "tokens_per_second": 10.0,
            "peak_vram_gb": 18.0,
        },
    )


@pytest.fixture
def faster_json(tmp_path):
    return _write_bench(
        tmp_path,
        "faster.json",
        {
            "model_name": "qwen3-14b",
            "load_seconds": 50.0,
            "total_seconds": 200.0,  # 75% faster
            "mean_latency_seconds": 25.0,
            "p50_latency_seconds": 22.0,
            "p90_latency_seconds": 35.0,
            "tokens_per_second": 40.0,  # 4x higher = FASTER
            "peak_vram_gb": 16.0,
        },
    )


def test_compare_flags_faster_when_total_drops(baseline_json, faster_json):
    buf = io.StringIO()
    with redirect_stdout(buf):
        compare(baseline_json, faster_json)
    out = buf.getvalue()
    assert "FASTER" in out
    # Specifically the total_seconds row must be flagged FASTER
    lines = [ln for ln in out.splitlines() if ln.startswith("total_seconds")]
    assert lines and "FASTER" in lines[0]


def test_compare_flags_slower_when_total_rises(baseline_json, tmp_path):
    slower = _write_bench(
        tmp_path,
        "slower.json",
        {
            "model_name": "qwen3-14b",
            "load_seconds": 70.0,
            "total_seconds": 1200.0,  # 50% slower
            "mean_latency_seconds": 150.0,
            "p50_latency_seconds": 135.0,
            "p90_latency_seconds": 200.0,
            "tokens_per_second": 6.0,  # lower = SLOWER
            "peak_vram_gb": 20.0,
        },
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        compare(baseline_json, slower)
    out = buf.getvalue()
    assert "SLOWER" in out
    lines = [ln for ln in out.splitlines() if ln.startswith("tokens_per_second")]
    assert lines and "SLOWER" in lines[0]


def test_compare_handles_different_models(baseline_json, tmp_path, caplog):
    other = _write_bench(
        tmp_path,
        "other_model.json",
        {
            "model_name": "phi-4",  # different model
            "load_seconds": 50.0,
            "total_seconds": 400.0,
            "mean_latency_seconds": 50.0,
            "p50_latency_seconds": 45.0,
            "p90_latency_seconds": 70.0,
            "tokens_per_second": 20.0,
            "peak_vram_gb": 12.0,
        },
    )
    import logging

    with caplog.at_level(logging.WARNING, logger="bench"):
        # Should warn but not crash
        buf = io.StringIO()
        with redirect_stdout(buf):
            compare(baseline_json, other)
    assert any("Comparing different models" in r.message for r in caplog.records)


def test_compare_nan_when_baseline_is_zero(tmp_path):
    """If a baseline value is 0, the percentage delta is NaN (not crash)."""
    baseline = _write_bench(
        tmp_path,
        "zero_baseline.json",
        {
            "model_name": "qwen3-14b",
            "load_seconds": 60.0,
            "total_seconds": 0.0,  # zero -> NaN delta
            "mean_latency_seconds": 0.0,
            "p50_latency_seconds": 0.0,
            "p90_latency_seconds": 0.0,
            "tokens_per_second": 0.0,
            "peak_vram_gb": 0.0,
        },
    )
    new = _write_bench(
        tmp_path,
        "new.json",
        {
            "model_name": "qwen3-14b",
            "load_seconds": 50.0,
            "total_seconds": 100.0,
            "mean_latency_seconds": 10.0,
            "p50_latency_seconds": 9.0,
            "p90_latency_seconds": 15.0,
            "tokens_per_second": 5.0,
            "peak_vram_gb": 16.0,
        },
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        compare(baseline, new)
    out = buf.getvalue()
    # Lines with bv=0 produce 'nan%' — should not crash
    assert "nan%" in out.lower()


# ----------------------------------------------------------------------
# run_benchmark — n_samples validation
# ----------------------------------------------------------------------


def test_run_benchmark_rejects_zero_samples():
    with pytest.raises(ValueError, match="n_samples must be >= 1"):
        run_benchmark("qwen3-14b", n_samples=0, max_new_tokens=128)


def test_run_benchmark_rejects_negative_samples():
    with pytest.raises(ValueError, match="n_samples must be >= 1"):
        run_benchmark("qwen3-14b", n_samples=-5, max_new_tokens=128)


# ----------------------------------------------------------------------
# Module-level: sys.path mutation guards
# ----------------------------------------------------------------------


def test_module_inserts_project_root_into_sys_path():
    """The module prepends the project root and src/ for imports.

    This is required so `python -m experiments.bench_inference` works
    even without PYTHONPATH set.
    """
    project_root = str(Path(__file__).parent.parent.resolve())
    assert project_root in sys.path or any(
        Path(p).resolve() == Path(project_root) for p in sys.path
    )
