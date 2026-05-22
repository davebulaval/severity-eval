"""Tests for severity_eval.metric (HuggingFace evaluate-compatible wrapper)."""

from __future__ import annotations

import pytest

from severity_eval.metric import _HF_AVAILABLE, compute_severity_metrics


PREDICTIONS = ["a", "b", "c", "d", "e", "wrong"]
REFERENCES = ["a", "b", "c", "d", "e", "f"]
SEVERITY = ["negligible"] * 5 + ["major"]
COSTS = [100, 1_000, 10_000, 100_000]


def test_returns_dict_with_core_keys():
    out = compute_severity_metrics(
        predictions=PREDICTIONS,
        references=REFERENCES,
        severity_annotations=SEVERITY,
        cost_levels=COSTS,
        n_sim=2_000,
        seed=0,
    )
    for key in (
        "accuracy",
        "error_rate",
        "expected_loss",
        "var",
        "tvar",
        "severity_profile",
        "cost_levels",
    ):
        assert key in out


def test_accuracy_matches_inputs():
    out = compute_severity_metrics(
        predictions=PREDICTIONS,
        references=REFERENCES,
        severity_annotations=SEVERITY,
        cost_levels=COSTS,
        n_sim=2_000,
        seed=0,
    )
    assert out["accuracy"] == pytest.approx(5 / 6)
    assert out["error_rate"] == pytest.approx(1 / 6)


def test_missing_cost_levels_raises():
    """Cost vector is mandatory; calling without it must fail loudly."""
    with pytest.raises(ValueError, match="cost_levels"):
        compute_severity_metrics(
            predictions=PREDICTIONS,
            references=REFERENCES,
            severity_annotations=SEVERITY,
        )


def test_reproducible_with_seed():
    a = compute_severity_metrics(
        predictions=PREDICTIONS,
        references=REFERENCES,
        severity_annotations=SEVERITY,
        cost_levels=COSTS,
        n_sim=1_000,
        seed=7,
    )
    b = compute_severity_metrics(
        predictions=PREDICTIONS,
        references=REFERENCES,
        severity_annotations=SEVERITY,
        cost_levels=COSTS,
        n_sim=1_000,
        seed=7,
    )
    assert a["expected_loss"] == b["expected_loss"]
    assert a["var"] == b["var"]


# ----------------------------------------------------------------------
# CompoundLossMetric (HuggingFace evaluate-compatible class)
# ----------------------------------------------------------------------

hf = pytest.importorskip("evaluate")


@pytest.mark.skipif(not _HF_AVAILABLE, reason="evaluate/datasets not installed")
def test_compound_loss_metric_info():
    """_info() returns a MetricInfo with the three expected feature keys."""
    from severity_eval.metric import CompoundLossMetric

    metric = CompoundLossMetric()
    info = metric._info()
    assert "compound-loss" in info.description.lower()
    assert set(info.features.keys()) == {
        "predictions",
        "references",
        "severity_annotations",
    }


@pytest.mark.skipif(not _HF_AVAILABLE, reason="evaluate/datasets not installed")
def test_compound_loss_metric_compute_returns_dict():
    """_compute returns a dict with the standard severity-report keys."""
    from severity_eval.metric import CompoundLossMetric

    metric = CompoundLossMetric()
    out = metric._compute(
        predictions=PREDICTIONS,
        references=REFERENCES,
        severity_annotations=SEVERITY,
        cost_levels=COSTS,
        n_sim=1_000,
        seed=0,
    )
    assert isinstance(out, dict)
    assert out["accuracy"] == pytest.approx(5 / 6)
    assert out["expected_loss"] > 0
    assert out["var"] >= out["expected_loss"] * 0.5  # rough sanity check
    assert len(out["severity_profile"]) == len(COSTS)


@pytest.mark.skipif(not _HF_AVAILABLE, reason="evaluate/datasets not installed")
def test_compound_loss_metric_missing_cost_levels_raises():
    """_compute without cost_levels must surface the same error as the wrapper."""
    from severity_eval.metric import CompoundLossMetric

    metric = CompoundLossMetric()
    with pytest.raises(ValueError, match="cost_levels"):
        metric._compute(
            predictions=PREDICTIONS,
            references=REFERENCES,
            severity_annotations=SEVERITY,
        )


@pytest.mark.skipif(not _HF_AVAILABLE, reason="evaluate/datasets not installed")
def test_compound_loss_metric_matches_direct_wrapper():
    """The class and the direct wrapper produce identical numerical results."""
    from severity_eval.metric import CompoundLossMetric

    metric = CompoundLossMetric()
    via_class = metric._compute(
        predictions=PREDICTIONS,
        references=REFERENCES,
        severity_annotations=SEVERITY,
        cost_levels=COSTS,
        n_sim=2_000,
        seed=11,
    )
    via_fn = compute_severity_metrics(
        predictions=PREDICTIONS,
        references=REFERENCES,
        severity_annotations=SEVERITY,
        cost_levels=COSTS,
        n_sim=2_000,
        seed=11,
    )
    assert via_class["expected_loss"] == via_fn["expected_loss"]
    assert via_class["var"] == via_fn["var"]
    assert via_class["accuracy"] == via_fn["accuracy"]


@pytest.mark.skipif(not _HF_AVAILABLE, reason="evaluate/datasets not installed")
def test_compound_loss_metric_propagates_alpha():
    """Different alpha levels produce different VaR / TVaR."""
    from severity_eval.metric import CompoundLossMetric

    metric = CompoundLossMetric()
    a_low = metric._compute(
        predictions=PREDICTIONS,
        references=REFERENCES,
        severity_annotations=SEVERITY,
        cost_levels=COSTS,
        n_sim=2_000,
        seed=0,
        alpha=0.50,
    )
    a_high = metric._compute(
        predictions=PREDICTIONS,
        references=REFERENCES,
        severity_annotations=SEVERITY,
        cost_levels=COSTS,
        n_sim=2_000,
        seed=0,
        alpha=0.99,
    )
    # The 99-th percentile must be >= the 50-th
    assert a_high["var"] >= a_low["var"]
    assert a_high["tvar"] >= a_low["tvar"]


@pytest.mark.skipif(not _HF_AVAILABLE, reason="evaluate/datasets not installed")
def test_compound_loss_metric_subclasses_evaluate_metric():
    """The class is a real evaluate.Metric (so HF's evaluate.load can find it)."""
    import evaluate as hf_evaluate

    from severity_eval.metric import CompoundLossMetric

    assert issubclass(CompoundLossMetric, hf_evaluate.Metric)
