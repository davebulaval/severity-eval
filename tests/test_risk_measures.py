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


# ----------------------------------------------------------------------
# wilson_score_interval -- binomial CI used for accuracy reporting.
# ----------------------------------------------------------------------


def test_wilson_score_centre_and_width_at_50_percent():
    """At p=0.5 the Wilson interval is symmetric around 0.5 and matches
    the textbook half-width 1.96 * sqrt(0.25/n) / (1 + 1.96^2/n) closely.
    """
    from severity_eval.risk_measures import wilson_score_interval

    lo, hi = wilson_score_interval(500, 1000)
    assert lo < 0.5 < hi
    assert (hi - lo) == pytest.approx(0.062, abs=0.005)  # ~6.2 percentage points


def test_wilson_score_zero_correct_stays_in_unit_interval():
    """At p_hat = 0 the normal approximation collapses to (0, 0). Wilson
    keeps the upper bound > 0 and the lower bound at 0.
    """
    from severity_eval.risk_measures import wilson_score_interval

    lo, hi = wilson_score_interval(0, 100)
    assert lo == 0.0
    assert 0.02 < hi < 0.05  # ~3.7% upper bound for 0/100


def test_wilson_score_all_correct_clamps_upper_to_one():
    """Symmetrically at p_hat = 1, the upper bound clamps to 1."""
    from severity_eval.risk_measures import wilson_score_interval

    lo, hi = wilson_score_interval(100, 100)
    assert hi == 1.0
    assert 0.95 < lo < 0.99


def test_wilson_score_narrows_with_more_data():
    """CI width should shrink as ~1/sqrt(n)."""
    from severity_eval.risk_measures import wilson_score_interval

    lo_s, hi_s = wilson_score_interval(50, 100)
    lo_l, hi_l = wilson_score_interval(500, 1000)
    assert (hi_l - lo_l) < (hi_s - lo_s)
    # ratio should be roughly sqrt(10) ~ 3.16
    ratio = (hi_s - lo_s) / (hi_l - lo_l)
    assert 2.5 < ratio < 4.0


def test_wilson_score_rejects_bad_inputs():
    from severity_eval.risk_measures import wilson_score_interval

    with pytest.raises(ValueError, match="n_total"):
        wilson_score_interval(5, 0)
    with pytest.raises(ValueError, match="n_success"):
        wilson_score_interval(15, 10)
    with pytest.raises(ValueError, match="alpha"):
        wilson_score_interval(5, 10, alpha=1.5)


# ----------------------------------------------------------------------
# paired_bootstrap_diff -- paired CI for model-vs-model comparison.
# ----------------------------------------------------------------------


def test_paired_bootstrap_zero_diff_when_vectors_identical():
    """L_A == L_B per-instance -> mean diff is 0 and CI brackets 0."""
    from severity_eval.risk_measures import paired_bootstrap_diff

    losses = np.array([100.0, 0.0, 50000.0, 0.0, 1000.0])
    mean_d, lo, hi = paired_bootstrap_diff(losses, losses, seed=0)
    assert mean_d == 0.0
    assert lo == 0.0 and hi == 0.0


def test_paired_bootstrap_recovers_constant_offset():
    """If model A is uniformly $100 worse per-error, mean diff = $100."""
    from severity_eval.risk_measures import paired_bootstrap_diff

    rng = np.random.default_rng(0)
    base = rng.choice([0.0, 100.0, 1000.0, 10000.0], size=500)
    a = base + 100.0
    b = base
    mean_d, lo, hi = paired_bootstrap_diff(a, b, n_bootstrap=300, seed=1)
    assert mean_d == pytest.approx(100.0)
    assert lo == pytest.approx(100.0)
    assert hi == pytest.approx(100.0)


def test_paired_bootstrap_picks_up_shared_difficulty():
    """Paired bootstrap should give tighter CI than unpaired when the
    two losses are correlated (shared instance difficulty).
    """
    from severity_eval.risk_measures import bootstrap_ci, paired_bootstrap_diff

    rng = np.random.default_rng(0)
    shared = rng.normal(0, 1000, size=400)  # per-instance difficulty
    a = shared + rng.normal(50, 10, size=400)
    b = shared + rng.normal(0, 10, size=400)
    # Paired diff = a - b ≈ normal(50, ~14): tight CI around 50
    mean_d, lo_p, hi_p = paired_bootstrap_diff(a, b, n_bootstrap=500, seed=1)
    # Unpaired CI on a and b separately, then diff of means: much wider
    a_lo, a_hi = bootstrap_ci(a, n_bootstrap=500, seed=2)
    b_lo, b_hi = bootstrap_ci(b, n_bootstrap=500, seed=3)
    paired_width = hi_p - lo_p
    unpaired_width = (a_hi - a_lo) + (b_hi - b_lo)
    assert paired_width < unpaired_width / 3, (
        f"paired CI ({paired_width:.1f}) must be much tighter than the sum "
        f"of unpaired CIs ({unpaired_width:.1f}) when losses share difficulty"
    )


def test_paired_bootstrap_rejects_misshapen_inputs():
    from severity_eval.risk_measures import paired_bootstrap_diff

    with pytest.raises(ValueError, match="must match"):
        paired_bootstrap_diff(np.zeros(10), np.zeros(11))
