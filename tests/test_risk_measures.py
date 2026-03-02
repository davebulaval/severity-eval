"""Tests for risk_measures module (T7-T10)."""

import numpy as np
import pytest

from severity_eval.compound_loss import simulate_aggregate_loss
from severity_eval.risk_measures import bootstrap_ci, compute_risk_measures

COST_LEVELS = np.array([100, 1000, 10000, 100000])
PI = np.array([0.4, 0.3, 0.2, 0.1])
N_QUERIES = 10000
P = 0.05
N_SIM = 100000
SEED = 42


@pytest.fixture
def losses():
    return simulate_aggregate_loss(N_QUERIES, P, COST_LEVELS, PI, N_SIM, seed=SEED)


# T7: VaR monotone in α
def test_var_monotone_in_alpha(losses):
    """VaR(0.99) ≥ VaR(0.95) ≥ VaR(0.90)."""
    var_90 = compute_risk_measures(losses, alpha=0.90)["var"]
    var_95 = compute_risk_measures(losses, alpha=0.95)["var"]
    var_99 = compute_risk_measures(losses, alpha=0.99)["var"]

    assert var_99 >= var_95 >= var_90


# T8: TVaR ≥ VaR
def test_tvar_geq_var(losses):
    """TVaR_α ≥ VaR_α for all α."""
    for alpha in [0.90, 0.95, 0.99]:
        m = compute_risk_measures(losses, alpha=alpha)
        assert m["tvar"] >= m["var"], f"TVaR < VaR at α={alpha}"


# T9: Bootstrap CI contains true value ~95% of time
def test_bootstrap_ci_coverage():
    """95% CI should contain the true E[S] most of the time."""
    mu_X = (COST_LEVELS * PI).sum()
    true_ES = N_QUERIES * P * mu_X

    n_trials = 20
    contains = 0
    for trial in range(n_trials):
        S = simulate_aggregate_loss(N_QUERIES, P, COST_LEVELS, PI, 50000, seed=trial)
        lo, hi = bootstrap_ci(S, statistic="expected_loss", confidence=0.95, n_bootstrap=500, seed=trial)
        if lo <= true_ES <= hi:
            contains += 1

    # With 20 trials at 95% CI, we expect ~19 hits; allow some slack
    assert contains >= 14, f"CI coverage {contains}/20 is too low"


# T10: Expected loss relationship with distribution shape
def test_expected_loss_vs_var(losses):
    """E[S] and VaR should be in reasonable relationship."""
    m = compute_risk_measures(losses, alpha=0.95)
    # VaR_95 should be greater than expected loss for right-skewed distribution
    assert m["var"] > m["expected_loss"]


def test_risk_measures_output_keys(losses):
    """Output should contain expected keys."""
    m = compute_risk_measures(losses)
    assert "expected_loss" in m
    assert "var" in m
    assert "tvar" in m


def test_invalid_alpha():
    """Should reject alpha outside (0, 1)."""
    S = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="alpha"):
        compute_risk_measures(S, alpha=1.5)
