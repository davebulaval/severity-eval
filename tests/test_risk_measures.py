"""Tests for severity_eval.risk_measures."""

from __future__ import annotations

import numpy as np
import pytest

from severity_eval.risk_measures import bootstrap_ci, compute_risk_measures


# ----------------------------------------------------------------------
# compute_risk_measures
# ----------------------------------------------------------------------


def test_expected_loss_equals_mean():
    rng = np.random.default_rng(0)
    losses = rng.uniform(0, 100, size=10_000)
    out = compute_risk_measures(losses, alpha=0.95)
    assert out["expected_loss"] == pytest.approx(losses.mean())


def test_var_equals_quantile():
    rng = np.random.default_rng(0)
    losses = rng.uniform(0, 100, size=10_000)
    out = compute_risk_measures(losses, alpha=0.95)
    assert out["var"] == pytest.approx(np.quantile(losses, 0.95))


def test_tvar_geq_var():
    """TVaR is the conditional mean of losses above VaR; must be >= VaR."""
    rng = np.random.default_rng(0)
    losses = rng.exponential(scale=100.0, size=20_000)
    out = compute_risk_measures(losses, alpha=0.95)
    assert out["tvar"] >= out["var"]


def test_tvar_strictly_above_var_when_continuous():
    """For a continuous distribution, TVaR is strictly > VaR."""
    rng = np.random.default_rng(0)
    losses = rng.exponential(scale=100.0, size=20_000)
    out = compute_risk_measures(losses, alpha=0.95)
    assert out["tvar"] > out["var"]


def test_zero_loss_array_returns_zeros():
    """All-zero losses: every measure is zero, including TVaR."""
    losses = np.zeros(1_000)
    out = compute_risk_measures(losses, alpha=0.95)
    assert out == {"expected_loss": 0.0, "var": 0.0, "tvar": 0.0}


def test_empty_returns_zeros_without_alpha_error():
    """Empty input degenerates to zeros."""
    out = compute_risk_measures(np.array([]), alpha=0.5)
    assert out == {"expected_loss": 0.0, "var": 0.0, "tvar": 0.0}


def test_invalid_alpha_low():
    with pytest.raises(ValueError, match="alpha must be in"):
        compute_risk_measures(np.array([1.0, 2.0]), alpha=0.0)


def test_invalid_alpha_high():
    with pytest.raises(ValueError, match="alpha must be in"):
        compute_risk_measures(np.array([1.0, 2.0]), alpha=1.0)


def test_invalid_alpha_validated_even_when_empty():
    """alpha is sanity-checked even for an empty input."""
    with pytest.raises(ValueError, match="alpha must be in"):
        compute_risk_measures(np.array([]), alpha=2.0)


def test_degenerate_tie_at_var():
    """When many points tie at the VaR (heavy point mass below alpha),
    TVaR should fall back to the upper tail including the quantile."""
    # 99% zeros, 1% at 100: VaR_0.95 = 0 exactly (quantile interpolation
    # lands on the all-zero region), TVaR uses >= to recover the 1.0 tail.
    losses = np.array([0.0] * 9_900 + [100.0] * 100)
    out = compute_risk_measures(losses, alpha=0.95)
    assert out["var"] == 0.0
    assert out["tvar"] > 0.0


# ----------------------------------------------------------------------
# bootstrap_ci
# ----------------------------------------------------------------------


def test_bootstrap_ci_brackets_mean():
    """The mean should fall inside its own bootstrap CI most of the time."""
    rng = np.random.default_rng(0)
    losses = rng.normal(loc=100.0, scale=10.0, size=2_000)
    lo, hi = bootstrap_ci(losses, statistic="expected_loss", n_bootstrap=500, seed=0)
    assert lo < losses.mean() < hi


def test_bootstrap_ci_reproducible_with_seed():
    rng = np.random.default_rng(0)
    losses = rng.normal(100, 10, size=1_000)
    a = bootstrap_ci(losses, n_bootstrap=200, seed=42)
    b = bootstrap_ci(losses, n_bootstrap=200, seed=42)
    assert a == b


def test_bootstrap_ci_varies_with_seed():
    """Different seeds produce different CIs (in general)."""
    rng = np.random.default_rng(0)
    losses = rng.normal(100, 10, size=1_000)
    a = bootstrap_ci(losses, n_bootstrap=200, seed=1)
    b = bootstrap_ci(losses, n_bootstrap=200, seed=2)
    assert a != b


def test_bootstrap_ci_var_statistic():
    rng = np.random.default_rng(0)
    losses = rng.exponential(100, size=2_000)
    lo, hi = bootstrap_ci(losses, statistic="var", alpha=0.95, n_bootstrap=300, seed=0)
    assert lo <= hi
    # 95-th percentile of an exp(100) is ~ -100*ln(0.05) = 300
    assert lo < 300 < hi or lo < np.quantile(losses, 0.95) < hi


def test_bootstrap_ci_tvar_statistic():
    rng = np.random.default_rng(0)
    losses = rng.exponential(100, size=2_000)
    lo, hi = bootstrap_ci(losses, statistic="tvar", alpha=0.95, n_bootstrap=300, seed=0)
    assert lo <= hi


def test_bootstrap_ci_unknown_statistic():
    with pytest.raises(ValueError, match="Unknown statistic"):
        bootstrap_ci(np.array([1.0, 2.0]), statistic="median")


def test_bootstrap_ci_invalid_alpha():
    with pytest.raises(ValueError, match="alpha must be in"):
        bootstrap_ci(np.array([1.0, 2.0]), alpha=-0.1)


def test_bootstrap_ci_invalid_n_bootstrap():
    with pytest.raises(ValueError, match="n_bootstrap must be"):
        bootstrap_ci(np.array([1.0, 2.0]), n_bootstrap=0)


def test_bootstrap_ci_empty_input():
    """Empty losses degenerate to (0, 0)."""
    assert bootstrap_ci(np.array([]), seed=0) == (0.0, 0.0)


def test_bootstrap_ci_narrows_with_more_data():
    """The CI for the mean should be tighter for larger samples."""
    rng = np.random.default_rng(0)
    small = rng.normal(100, 10, size=50)
    large = rng.normal(100, 10, size=5_000)
    s_lo, s_hi = bootstrap_ci(small, n_bootstrap=300, seed=1)
    l_lo, l_hi = bootstrap_ci(large, n_bootstrap=300, seed=1)
    assert (l_hi - l_lo) < (s_hi - s_lo)
