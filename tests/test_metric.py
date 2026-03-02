"""Tests for metric module (T17-T19)."""

import importlib.util

import pytest

from severity_eval.metric import compute_severity_metrics

HF_AVAILABLE = importlib.util.find_spec("evaluate") is not None

PREDICTIONS = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
REFERENCES = ["a", "b", "X", "d", "e", "Y", "g", "h", "Z", "j"]
SEVERITY = [
    "negligible",
    "negligible",
    "minor",
    "negligible",
    "negligible",
    "major",
    "negligible",
    "negligible",
    "critical",
    "negligible",
]
COST_LEVELS = [100, 1000, 10000, 100000]


# T17: Compatible evaluate.Metric interface
@pytest.mark.skipif(not HF_AVAILABLE, reason="evaluate not installed")
def test_huggingface_evaluate_interface():
    from severity_eval.metric import CompoundLossMetric

    metric = CompoundLossMetric()
    info = metric._info()
    assert info is not None
    assert info.description is not None


# T18: Output contains all expected keys
def test_output_keys():
    result = compute_severity_metrics(
        predictions=PREDICTIONS,
        references=REFERENCES,
        severity_annotations=SEVERITY,
        cost_levels=COST_LEVELS,
        n_sim=10000,
        seed=42,
    )
    for key in [
        "accuracy",
        "error_rate",
        "n_samples",
        "n_errors",
        "severity_profile",
        "expected_loss",
        "var",
        "tvar",
        "cost_levels",
    ]:
        assert key in result, f"Missing key: {key}"


def test_accuracy_computation():
    result = compute_severity_metrics(
        predictions=PREDICTIONS,
        references=REFERENCES,
        severity_annotations=SEVERITY,
        cost_levels=COST_LEVELS,
        n_sim=10000,
        seed=42,
    )
    assert result["accuracy"] == pytest.approx(0.7)
    assert result["error_rate"] == pytest.approx(0.3)
    assert result["n_samples"] == 10
    assert result["n_errors"] == 3


def test_severity_profile_from_errors():
    result = compute_severity_metrics(
        predictions=PREDICTIONS,
        references=REFERENCES,
        severity_annotations=SEVERITY,
        cost_levels=COST_LEVELS,
        n_sim=10000,
        seed=42,
    )
    pi = result["severity_profile"]
    assert pi[0] == pytest.approx(0.0)
    assert pi[1] == pytest.approx(1 / 3, abs=0.01)
    assert pi[2] == pytest.approx(1 / 3, abs=0.01)
    assert pi[3] == pytest.approx(1 / 3, abs=0.01)


# T19: Consistency
def test_expected_loss_positive_when_errors():
    result = compute_severity_metrics(
        predictions=PREDICTIONS,
        references=REFERENCES,
        severity_annotations=SEVERITY,
        cost_levels=COST_LEVELS,
        n_sim=50000,
        seed=42,
    )
    assert result["expected_loss"] > 0
    assert result["var"] > 0
    assert result["tvar"] >= result["var"]


def test_perfect_predictions_zero_loss():
    result = compute_severity_metrics(
        predictions=["a", "b", "c"],
        references=["a", "b", "c"],
        severity_annotations=["negligible", "negligible", "negligible"],
        cost_levels=COST_LEVELS,
        n_sim=10000,
        seed=42,
    )
    assert result["accuracy"] == 1.0
    assert result["expected_loss"] == 0.0


def test_different_costs_different_loss():
    r_low = compute_severity_metrics(
        predictions=PREDICTIONS,
        references=REFERENCES,
        severity_annotations=SEVERITY,
        cost_levels=[100, 1000, 10000, 100000],
        n_sim=50000,
        seed=42,
    )
    r_high = compute_severity_metrics(
        predictions=PREDICTIONS,
        references=REFERENCES,
        severity_annotations=SEVERITY,
        cost_levels=[500, 5000, 50000, 500000],
        n_sim=50000,
        seed=42,
    )
    assert r_high["expected_loss"] > r_low["expected_loss"]


def test_cost_levels_required():
    """compute_severity_metrics should raise if cost_levels is None."""
    with pytest.raises(ValueError, match="cost_levels is required"):
        compute_severity_metrics(
            predictions=PREDICTIONS,
            references=REFERENCES,
            severity_annotations=SEVERITY,
        )
