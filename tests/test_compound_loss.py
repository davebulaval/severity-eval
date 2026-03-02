"""Tests for compound_loss module (T1-T6)."""

import numpy as np
import pytest

from severity_eval.compound_loss import simulate_aggregate_loss

# Shared fixtures
COST_LEVELS = np.array([100, 1000, 10000, 100000])
PI = np.array([0.4, 0.3, 0.2, 0.1])
N_QUERIES = 10000
P = 0.05
N_SIM = 200000
SEED = 42


# T1: E[S] converges to np*μ_X (law of large numbers)
def test_expected_loss_converges_to_theoretical():
    """E[S] from MC should be within 1% of np*μ_X for large n_sim."""
    mu_X = (COST_LEVELS * PI).sum()
    theoretical_ES = N_QUERIES * P * mu_X

    S = simulate_aggregate_loss(N_QUERIES, P, COST_LEVELS, PI, N_SIM, seed=SEED)
    mc_ES = S.mean()

    relative_error = abs(mc_ES - theoretical_ES) / theoretical_ES
    assert relative_error < 0.01, f"Relative error {relative_error:.4f} > 1%"


# T2: Var(S) converges to theoretical formula
def test_variance_converges_to_theoretical():
    """Var(S) from MC ~ np*σ²_X + np(1-p)*μ²_X."""
    mu_X = (COST_LEVELS * PI).sum()
    E_X2 = (COST_LEVELS**2 * PI).sum()
    sigma2_X = E_X2 - mu_X**2

    # Var(S) = n*p*σ²_X + n*p*(1-p)*μ²_X
    theoretical_var = N_QUERIES * P * sigma2_X + N_QUERIES * P * (1 - P) * mu_X**2

    S = simulate_aggregate_loss(N_QUERIES, P, COST_LEVELS, PI, N_SIM, seed=SEED)
    mc_var = S.var()

    relative_error = abs(mc_var - theoretical_var) / theoretical_var
    assert relative_error < 0.02, f"Relative error {relative_error:.4f} > 2%"


# T3: S ≥ 0 always
def test_aggregate_loss_non_negative():
    """All simulated S values should be ≥ 0."""
    S = simulate_aggregate_loss(N_QUERIES, P, COST_LEVELS, PI, N_SIM, seed=SEED)
    assert (S >= 0).all()


# T4: p=0 → S=0
def test_zero_error_rate_gives_zero_loss():
    """If p=0, all losses should be zero."""
    S = simulate_aggregate_loss(N_QUERIES, 0.0, COST_LEVELS, PI, N_SIM, seed=SEED)
    assert (S == 0).all()


# T5: p=1 → S = n*μ_X on average
def test_certain_errors():
    """If p=1, E[S] = n * μ_X."""
    n = 100  # small n for speed with p=1
    mu_X = (COST_LEVELS * PI).sum()
    theoretical_ES = n * mu_X

    S = simulate_aggregate_loss(n, 1.0, COST_LEVELS, PI, N_SIM, seed=SEED)
    mc_ES = S.mean()

    relative_error = abs(mc_ES - theoretical_ES) / theoretical_ES
    assert relative_error < 0.01, f"Relative error {relative_error:.4f} > 1%"


# T6: Reproducibility with seed
def test_reproducibility_with_seed():
    """Same seed → same results."""
    S1 = simulate_aggregate_loss(N_QUERIES, P, COST_LEVELS, PI, 1000, seed=42)
    S2 = simulate_aggregate_loss(N_QUERIES, P, COST_LEVELS, PI, 1000, seed=42)
    np.testing.assert_array_equal(S1, S2)


def test_different_seeds_give_different_results():
    """Different seeds → different results."""
    S1 = simulate_aggregate_loss(N_QUERIES, P, COST_LEVELS, PI, 1000, seed=42)
    S2 = simulate_aggregate_loss(N_QUERIES, P, COST_LEVELS, PI, 1000, seed=99)
    assert not np.array_equal(S1, S2)


def test_invalid_severity_profile_sum():
    """Should reject severity profile that doesn't sum to 1."""
    bad_pi = np.array([0.5, 0.3, 0.1, 0.05])
    with pytest.raises(ValueError, match="must sum to 1"):
        simulate_aggregate_loss(N_QUERIES, P, COST_LEVELS, bad_pi, 100, seed=SEED)


def test_invalid_error_rate():
    """Should reject error rate outside [0, 1]."""
    with pytest.raises(ValueError, match="error_rate"):
        simulate_aggregate_loss(N_QUERIES, 1.5, COST_LEVELS, PI, 100, seed=SEED)
