"""Tests for severity profile validation and normalization."""

import numpy as np
import pytest

from severity_eval.validation import validate_severity_profile


def test_valid_probabilities():
    """Valid π summing to 1 should pass through unchanged."""
    pi = validate_severity_profile([0.4, 0.3, 0.2, 0.1])
    np.testing.assert_allclose(pi, [0.4, 0.3, 0.2, 0.1])
    assert isinstance(pi, np.ndarray)


def test_valid_probabilities_numpy():
    """Works with numpy arrays."""
    pi = validate_severity_profile(np.array([0.5, 0.5]))
    np.testing.assert_allclose(pi, [0.5, 0.5])


def test_percentages_auto_converted():
    """π given as percentages (sum=100) should auto-convert to probabilities."""
    pi = validate_severity_profile([40, 30, 20, 10])
    np.testing.assert_allclose(pi, [0.4, 0.3, 0.2, 0.1])


def test_percentages_exact_100():
    """Percentages summing to exactly 100."""
    pi = validate_severity_profile([25, 25, 25, 25])
    np.testing.assert_allclose(pi, [0.25, 0.25, 0.25, 0.25])


def test_percentages_close_to_100():
    """Percentages summing to ~99.9 or ~100.1 should still convert."""
    pi = validate_severity_profile([40.1, 29.9, 20, 10])
    assert np.isclose(pi.sum(), 1.0)


def test_rejects_negative():
    """Negative values should raise."""
    with pytest.raises(ValueError, match="negative"):
        validate_severity_profile([0.5, -0.1, 0.6])


def test_rejects_bad_sum():
    """Values not summing to 1 or 100 should raise."""
    with pytest.raises(ValueError, match="must sum to 1"):
        validate_severity_profile([0.5, 0.3, 0.1, 0.05])


def test_rejects_sum_50():
    """Sum of 50 is neither probabilities nor percentages."""
    with pytest.raises(ValueError, match="must sum to 1"):
        validate_severity_profile([20, 15, 10, 5])


def test_rejects_empty():
    """Empty array should raise."""
    with pytest.raises(ValueError, match="non-empty"):
        validate_severity_profile([])


def test_rejects_2d():
    """2-D array should raise."""
    with pytest.raises(ValueError, match="1-D"):
        validate_severity_profile([[0.5, 0.5], [0.3, 0.7]])


def test_three_levels():
    """Works with any number of levels."""
    pi = validate_severity_profile([50, 30, 20])
    np.testing.assert_allclose(pi, [0.5, 0.3, 0.2])


def test_six_levels():
    """Works with 6 levels too."""
    pi = validate_severity_profile([0.1, 0.1, 0.2, 0.2, 0.2, 0.2])
    assert np.isclose(pi.sum(), 1.0)


def test_custom_name_in_error():
    """Custom name should appear in error message."""
    with pytest.raises(ValueError, match="my_probs"):
        validate_severity_profile([0.5], name="my_probs")


def test_passes_through_simulate():
    """End-to-end: percentages should work in simulate_aggregate_loss."""
    from severity_eval.compound_loss import simulate_aggregate_loss

    S = simulate_aggregate_loss(
        n_queries=100,
        error_rate=0.1,
        cost_levels=[100, 1000, 10000, 100000],
        severity_profile=[40, 30, 20, 10],  # percentages
        n_sim=100,
        seed=42,
    )
    assert len(S) == 100
    assert (S >= 0).all()


def test_passes_through_routing():
    """End-to-end: percentages should work in analyze_routing."""
    from severity_eval.routing import analyze_routing

    result = analyze_routing(
        n_queries=1000,
        error_rate=0.05,
        cost_levels=[100, 1000, 10000, 100000],
        severity_profile=[40, 30, 20, 10],  # percentages
        retention_threshold=10000,
    )
    assert result.routing_ratio == pytest.approx(0.3)


def test_passes_through_ruin():
    """End-to-end: percentages should work in compute_lundberg_R."""
    from severity_eval.ruin import compute_lundberg_R

    R = compute_lundberg_R(
        claim_rate=500,
        cost_levels=[100, 1000, 10000, 100000],
        severity_profile=[40, 30, 20, 10],  # percentages
        premium_rate=500 * 12340 * 1.02,
    )
    assert R > 0
