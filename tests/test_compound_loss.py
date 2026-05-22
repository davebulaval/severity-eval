"""Tests for severity_eval.compound_loss."""

from __future__ import annotations

import numpy as np
import pytest

from severity_eval.compound_loss import simulate_aggregate_loss


COSTS = np.array([100.0, 1_000.0, 10_000.0, 100_000.0])
PI = np.array([0.4, 0.3, 0.2, 0.1])


def test_output_shape():
    S = simulate_aggregate_loss(1_000, 0.1, COSTS, PI, n_sim=500, seed=0)
    assert S.shape == (500,)


def test_zero_error_rate_returns_zeros():
    """No errors → S = 0 everywhere."""
    S = simulate_aggregate_loss(10_000, 0.0, COSTS, PI, n_sim=300, seed=0)
    assert np.all(S == 0.0)


def test_zero_queries_returns_zeros():
    """No queries → no errors → S = 0."""
    S = simulate_aggregate_loss(0, 0.5, COSTS, PI, n_sim=300, seed=0)
    assert np.all(S == 0.0)


def test_unit_error_rate_S_is_sum_of_n_severities():
    """If p=1, every query is an error and S equals the sum of n IID severities."""
    n = 50
    S = simulate_aggregate_loss(n, 1.0, COSTS, PI, n_sim=2_000, seed=0)
    # All n severities are drawn IID from (COSTS, PI)
    expected_mean = n * float((COSTS * PI).sum())
    assert S.mean() == pytest.approx(expected_mean, rel=0.05)
    # Lower bound: smallest possible S is n * c_min
    assert S.min() >= n * COSTS.min()
    # Upper bound: largest possible S is n * c_max
    assert S.max() <= n * COSTS.max()


def test_mean_matches_closed_form():
    """E[S] = n p mu_X to within MC noise."""
    n, p = 5_000, 0.05
    mu_X = float((COSTS * PI).sum())
    S = simulate_aggregate_loss(n, p, COSTS, PI, n_sim=5_000, seed=42)
    expected = n * p * mu_X
    # Standard error of the mean of S is sqrt(Var(S)/n_sim); allow 5% relative.
    assert S.mean() == pytest.approx(expected, rel=0.05)


def test_variance_matches_closed_form():
    """Var(S) = n p sigma_X^2 + n p (1-p) mu_X^2."""
    n, p = 2_000, 0.10
    mu_X = float((COSTS * PI).sum())
    sigma_X2 = float((COSTS**2 * PI).sum() - mu_X**2)
    S = simulate_aggregate_loss(n, p, COSTS, PI, n_sim=10_000, seed=42)
    expected = n * p * sigma_X2 + n * p * (1 - p) * mu_X**2
    assert S.var() == pytest.approx(expected, rel=0.10)


def test_seed_reproducibility():
    a = simulate_aggregate_loss(1_000, 0.1, COSTS, PI, n_sim=300, seed=123)
    b = simulate_aggregate_loss(1_000, 0.1, COSTS, PI, n_sim=300, seed=123)
    np.testing.assert_array_equal(a, b)


def test_seed_variation():
    a = simulate_aggregate_loss(1_000, 0.1, COSTS, PI, n_sim=300, seed=1)
    b = simulate_aggregate_loss(1_000, 0.1, COSTS, PI, n_sim=300, seed=2)
    assert not np.array_equal(a, b)


def test_invalid_error_rate():
    for bad in [-0.1, 1.1]:
        with pytest.raises(ValueError, match="error_rate"):
            simulate_aggregate_loss(100, bad, COSTS, PI, n_sim=10)


def test_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        simulate_aggregate_loss(100, 0.1, np.array([100.0, 1_000.0]), PI, n_sim=10)


def test_invalid_profile_propagates():
    """validation.validate_severity_profile is called → its errors bubble up."""
    with pytest.raises(ValueError, match="must sum to 1"):
        simulate_aggregate_loss(100, 0.1, COSTS, [0.5, 0.0, 0.0, 0.2])


def test_severities_only_use_supplied_costs():
    """Every realised non-zero loss must decompose into multiples of c_k."""
    n = 5
    S = simulate_aggregate_loss(n, 1.0, COSTS, PI, n_sim=1_000, seed=0)
    # With n=5, every S is a sum of 5 cost-values; smallest unit is gcd.
    # We test the weaker property: S/100 is integer (since 100 divides all costs)
    np.testing.assert_allclose(S / 100, np.round(S / 100))


def test_low_p_large_n_sim_does_not_blow_memory():
    """Stress-test the chunking logic: large n_queries but tiny p stays bounded."""
    S = simulate_aggregate_loss(
        n_queries=100_000,
        error_rate=1e-4,
        cost_levels=COSTS,
        severity_profile=PI,
        n_sim=2_000,
        seed=0,
    )
    assert S.shape == (2_000,)
    # E[N] = 10 → most S values are small, never overflow
    assert np.isfinite(S).all()
