"""Tests for the public API (severity_eval.evaluate → SeverityReport)."""

import numpy as np
import pytest

import severity_eval
from severity_eval.api import SeverityReport, evaluate

# Fixture data: 10 predictions, 3 errors
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


@pytest.fixture
def report():
    return evaluate(
        predictions=PREDICTIONS,
        references=REFERENCES,
        severity_annotations=SEVERITY,
        cost_levels=COST_LEVELS,
        n_sim=10000,
        seed=42,
    )


# ------------------------------------------------------------------
# Return type and basic attributes
# ------------------------------------------------------------------


def test_returns_severity_report(report):
    assert isinstance(report, SeverityReport)


def test_accuracy(report):
    assert report.accuracy == pytest.approx(0.7)
    assert report.error_rate == pytest.approx(0.3)
    assert report.n_samples == 10
    assert report.n_errors == 3


def test_severity_profile(report):
    pi = report.severity_profile
    assert len(pi) == 4
    assert pi[0] == pytest.approx(0.0)
    assert pi[1] == pytest.approx(1 / 3, abs=0.01)
    assert pi[2] == pytest.approx(1 / 3, abs=0.01)
    assert pi[3] == pytest.approx(1 / 3, abs=0.01)


def test_cost_levels_stored(report):
    np.testing.assert_array_equal(report.cost_levels, COST_LEVELS)
    assert report.labels == ["negligible", "minor", "major", "critical"]


def test_risk_measures_populated(report):
    assert report.expected_loss > 0
    assert report.var > 0
    assert report.tvar >= report.var


def test_losses_array(report):
    assert isinstance(report.losses, np.ndarray)
    assert len(report.losses) == 10000
    assert (report.losses >= 0).all()


# ------------------------------------------------------------------
# Output formats
# ------------------------------------------------------------------


def test_summary_string(report):
    s = report.summary()
    assert "Severity Report" in s
    assert "E[S]" in s
    assert "VaR" in s
    assert "pi_k" in s
    assert "c_k" in s
    assert "mu_X" in s


def test_str_is_summary(report):
    assert str(report) == report.summary()


def test_to_dict(report):
    d = report.to_dict()
    assert isinstance(d, dict)
    assert d["cost_levels"] == [100, 1000, 10000, 100000]
    assert d["accuracy"] == pytest.approx(0.7)
    assert d["expected_loss"] > 0
    assert isinstance(d["severity_profile"], list)
    assert len(d["severity_profile"]) == 4


def test_to_dataframe(report):
    dfs = report.to_dataframe()
    assert "summary" in dfs
    assert "severity_profile" in dfs
    assert len(dfs["severity_profile"]) == 4
    assert "E[S]" in dfs["summary"].columns


def test_to_latex(report):
    tex = report.to_latex()
    assert r"\begin{tabular}" in tex
    assert r"\toprule" in tex
    assert "E[S]" in tex
    assert "c_k" in tex


# ------------------------------------------------------------------
# Routing integration
# ------------------------------------------------------------------


def test_routing_included_when_threshold_set():
    report = evaluate(
        predictions=PREDICTIONS,
        references=REFERENCES,
        severity_annotations=SEVERITY,
        cost_levels=COST_LEVELS,
        n_sim=10000,
        seed=42,
        routing_threshold=10000,
    )
    assert report.routing is not None
    assert report.routing.cost_reduction_pct > 0
    assert "routing" in report.summary().lower()


def test_no_routing_by_default(report):
    assert report.routing is None


# ------------------------------------------------------------------
# Top-level import
# ------------------------------------------------------------------


def test_importable_from_top_level():
    assert hasattr(severity_eval, "evaluate")
    assert hasattr(severity_eval, "SeverityReport")


# ------------------------------------------------------------------
# Perfect predictions
# ------------------------------------------------------------------


def test_perfect_predictions():
    report = evaluate(
        predictions=["a", "b", "c"],
        references=["a", "b", "c"],
        severity_annotations=["negligible", "minor", "major"],
        cost_levels=COST_LEVELS,
        n_sim=1000,
    )
    assert report.accuracy == 1.0
    assert report.expected_loss == 0.0
    assert report.n_errors == 0


# ------------------------------------------------------------------
# Different cost levels → different E[S]
# ------------------------------------------------------------------


def test_cost_levels_affect_expected_loss():
    r_low = evaluate(
        predictions=PREDICTIONS,
        references=REFERENCES,
        severity_annotations=SEVERITY,
        cost_levels=[100, 1000, 10000, 100000],
        n_sim=50000,
        seed=42,
    )
    r_high = evaluate(
        predictions=PREDICTIONS,
        references=REFERENCES,
        severity_annotations=SEVERITY,
        cost_levels=[500, 5000, 50000, 500000],
        n_sim=50000,
        seed=42,
    )
    assert r_high.expected_loss > r_low.expected_loss


# ------------------------------------------------------------------
# Custom labels
# ------------------------------------------------------------------


def test_custom_labels():
    report = evaluate(
        predictions=["a", "b", "wrong"],
        references=["a", "b", "c"],
        severity_annotations=["low", "low", "high"],
        cost_levels=[100, 10000],
        labels=["low", "high"],
        n_sim=1000,
    )
    assert report.labels == ["low", "high"]
    assert report.n_errors == 1
    assert report.severity_profile[1] == pytest.approx(1.0)


def test_arbitrary_k_levels():
    """Should work with any number of levels, auto-generating labels."""
    report = evaluate(
        predictions=["a", "b", "wrong1", "wrong2"],
        references=["a", "b", "c", "d"],
        severity_annotations=["level_1", "level_2", "level_3", "level_2"],
        cost_levels=[10, 100, 1000],
        labels=["level_1", "level_2", "level_3"],
        n_sim=1000,
    )
    assert len(report.cost_levels) == 3
    assert report.labels == ["level_1", "level_2", "level_3"]
    assert report.n_errors == 2


def test_auto_labels_non_4_levels():
    """When K != 4 and no labels, auto-generate level_1, level_2, ..."""
    report = evaluate(
        predictions=["a", "wrong"],
        references=["a", "b"],
        severity_annotations=["level_1", "level_2"],
        cost_levels=[50, 500],
        n_sim=1000,
    )
    assert report.labels == ["level_1", "level_2"]


def test_mismatched_labels_raises():
    with pytest.raises(ValueError, match="len\\(labels\\)"):
        evaluate(
            predictions=["a"],
            references=["b"],
            severity_annotations=["x"],
            cost_levels=[100, 1000],
            labels=["x", "y", "z"],
        )


def test_unknown_annotation_raises():
    with pytest.raises(ValueError, match="Unknown severity label"):
        evaluate(
            predictions=["a", "wrong"],
            references=["a", "b"],
            severity_annotations=["negligible", "INVALID"],
            cost_levels=COST_LEVELS,
        )


# ------------------------------------------------------------------
# Backward compatibility: metric.compute_severity_metrics
# ------------------------------------------------------------------


def test_metric_backward_compat():
    from severity_eval.metric import compute_severity_metrics

    d = compute_severity_metrics(
        predictions=PREDICTIONS,
        references=REFERENCES,
        severity_annotations=SEVERITY,
        cost_levels=[100, 1000, 10000, 100000],
        n_sim=10000,
        seed=42,
    )
    assert isinstance(d, dict)
    assert d["accuracy"] == pytest.approx(0.7)
    assert d["expected_loss"] > 0
