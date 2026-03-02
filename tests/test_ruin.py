"""Tests for ruin module (T11-T13)."""

import numpy as np
import pytest

from severity_eval.ruin import (
    compute_lundberg_R,
    compute_reserve,
    ruin_probability_curve,
    simulate_ruin_probability,
)

COST_LEVELS = np.array([100, 1000, 10000, 100000])
PI = np.array([0.4, 0.3, 0.2, 0.1])
N_QUERIES = 10000
P = 0.05
THETA = 0.02  # safety loading

CLAIM_RATE = N_QUERIES * P  # λ = 500
MU_X = (COST_LEVELS * PI).sum()
PREMIUM_RATE = (1 + THETA) * CLAIM_RATE * MU_X


# T11: R > 0 when premium > E[S]
def test_lundberg_R_positive():
    """R should be positive when premium > expected loss."""
    R = compute_lundberg_R(CLAIM_RATE, COST_LEVELS, PI, PREMIUM_RATE)
    assert R > 0, f"R = {R} should be positive"


# T12: u* increases with heavier tail (smaller R)
def test_reserve_increases_with_heavier_tail():
    """Heavier tail → smaller R → larger u*."""
    # Light tail: concentrated on low costs
    pi_light = np.array([0.70, 0.20, 0.08, 0.02])
    mu_light = (COST_LEVELS * pi_light).sum()
    prem_light = (1 + THETA) * CLAIM_RATE * mu_light
    R_light = compute_lundberg_R(CLAIM_RATE, COST_LEVELS, pi_light, prem_light)
    u_light = compute_reserve(R_light)

    # Heavy tail: more weight on high costs
    pi_heavy = np.array([0.20, 0.20, 0.30, 0.30])
    mu_heavy = (COST_LEVELS * pi_heavy).sum()
    prem_heavy = (1 + THETA) * CLAIM_RATE * mu_heavy
    R_heavy = compute_lundberg_R(CLAIM_RATE, COST_LEVELS, pi_heavy, prem_heavy)
    u_heavy = compute_reserve(R_heavy)

    assert R_light > R_heavy, f"R_light={R_light} should be > R_heavy={R_heavy}"
    assert u_heavy > u_light, f"u_heavy={u_heavy} should be > u_light={u_light}"


# T13: Lundberg bound holds: ψ(u) ≤ exp(-Ru) verified by MC
def test_lundberg_bound_holds():
    """MC ruin probability should be ≤ Lundberg bound for u ≥ u*."""
    R = compute_lundberg_R(CLAIM_RATE, COST_LEVELS, PI, PREMIUM_RATE)
    u_star = compute_reserve(R, ruin_target=0.01)

    # MC ruin estimation
    rng = np.random.default_rng(42)
    n_mc = 10000
    n_periods = 10
    ruin_count = 0

    for _ in range(n_mc):
        U = u_star
        ruined = False
        for _ in range(n_periods):
            N_t = rng.poisson(CLAIM_RATE)
            if N_t > 0:
                X_t = rng.choice(COST_LEVELS, size=N_t, p=PI).sum()
            else:
                X_t = 0
            U = U + PREMIUM_RATE - X_t
            if U < 0:
                ruined = True
                break
        if ruined:
            ruin_count += 1

    psi_mc = ruin_count / n_mc
    lundberg_bound = np.exp(-R * u_star)

    # Allow 50% slack for MC variance
    assert psi_mc <= lundberg_bound * 1.5, f"MC ruin prob {psi_mc:.4f} > 1.5 * Lundberg bound {lundberg_bound:.4f}"


def test_compute_reserve_values():
    """Reserve should be positive and reasonable."""
    R = compute_lundberg_R(CLAIM_RATE, COST_LEVELS, PI, PREMIUM_RATE)
    u = compute_reserve(R, ruin_target=0.01)
    assert u > 0
    # More conservative target → larger reserve
    u_conservative = compute_reserve(R, ruin_target=0.001)
    assert u_conservative > u


def test_invalid_R():
    """Should reject non-positive R."""
    with pytest.raises(ValueError, match="positive"):
        compute_reserve(-1.0)


# ------------------------------------------------------------------
# MC ruin probability: simulate_ruin_probability
# ------------------------------------------------------------------


def test_simulate_ruin_probability_returns_float():
    """ψ_MC(u) should be a float in [0, 1]."""
    psi = simulate_ruin_probability(
        initial_reserve=500000,
        claim_rate=CLAIM_RATE,
        cost_levels=COST_LEVELS,
        severity_profile=PI,
        premium_rate=PREMIUM_RATE,
        n_sim=1000,
        n_periods=10,
        seed=42,
    )
    assert isinstance(psi, float)
    assert 0.0 <= psi <= 1.0


def test_simulate_ruin_probability_zero_reserve():
    """With u=0, ruin probability should be high."""
    psi = simulate_ruin_probability(
        initial_reserve=0,
        claim_rate=CLAIM_RATE,
        cost_levels=COST_LEVELS,
        severity_profile=PI,
        premium_rate=PREMIUM_RATE,
        n_sim=5000,
        n_periods=20,
        seed=42,
    )
    assert psi > 0.1, f"ψ(0) = {psi} should be high"


def test_simulate_ruin_probability_huge_reserve():
    """With very large u, ruin probability should be ~0."""
    psi = simulate_ruin_probability(
        initial_reserve=1e12,
        claim_rate=CLAIM_RATE,
        cost_levels=COST_LEVELS,
        severity_profile=PI,
        premium_rate=PREMIUM_RATE,
        n_sim=1000,
        n_periods=10,
        seed=42,
    )
    assert psi == 0.0


def test_simulate_ruin_leq_lundberg_bound():
    """ψ_MC(u*) should be ≤ Lundberg bound (with MC slack)."""
    R = compute_lundberg_R(CLAIM_RATE, COST_LEVELS, PI, PREMIUM_RATE)
    u_star = compute_reserve(R, ruin_target=0.01)
    psi = simulate_ruin_probability(
        initial_reserve=u_star,
        claim_rate=CLAIM_RATE,
        cost_levels=COST_LEVELS,
        severity_profile=PI,
        premium_rate=PREMIUM_RATE,
        n_sim=10000,
        n_periods=10,
        seed=42,
    )
    lundberg_bound = np.exp(-R * u_star)
    assert psi <= lundberg_bound * 1.5, f"ψ_MC={psi:.4f} > 1.5 * bound={lundberg_bound:.4f}"


def test_simulate_ruin_monotone_in_reserve():
    """ψ(u1) > ψ(u2) when u1 < u2."""
    psi_low = simulate_ruin_probability(
        initial_reserve=100000,
        claim_rate=CLAIM_RATE,
        cost_levels=COST_LEVELS,
        severity_profile=PI,
        premium_rate=PREMIUM_RATE,
        n_sim=5000,
        n_periods=20,
        seed=42,
    )
    psi_high = simulate_ruin_probability(
        initial_reserve=1000000,
        claim_rate=CLAIM_RATE,
        cost_levels=COST_LEVELS,
        severity_profile=PI,
        premium_rate=PREMIUM_RATE,
        n_sim=5000,
        n_periods=20,
        seed=42,
    )
    assert psi_low >= psi_high


# ------------------------------------------------------------------
# Ruin probability curve
# ------------------------------------------------------------------


def test_ruin_curve_shape():
    """Curve should return arrays of matching length."""
    u_values, psi_mc, psi_bound = ruin_probability_curve(
        u_range=np.linspace(10000, 500000, 5),
        claim_rate=CLAIM_RATE,
        cost_levels=COST_LEVELS,
        severity_profile=PI,
        premium_rate=PREMIUM_RATE,
        n_sim=1000,
        n_periods=10,
        seed=42,
    )
    assert len(u_values) == 5
    assert len(psi_mc) == 5
    assert len(psi_bound) == 5


def test_ruin_curve_bound_decreasing():
    """Lundberg bound should be strictly decreasing."""
    u_range = np.linspace(10000, 500000, 10)
    _, _, psi_bound = ruin_probability_curve(
        u_range=u_range,
        claim_rate=CLAIM_RATE,
        cost_levels=COST_LEVELS,
        severity_profile=PI,
        premium_rate=PREMIUM_RATE,
        n_sim=500,
        n_periods=5,
        seed=42,
    )
    for i in range(len(psi_bound) - 1):
        assert psi_bound[i] > psi_bound[i + 1]


def test_ruin_curve_mc_leq_bound():
    """MC estimates should generally be ≤ Lundberg bound."""
    u_range = np.linspace(100000, 1000000, 5)
    _, psi_mc, psi_bound = ruin_probability_curve(
        u_range=u_range,
        claim_rate=CLAIM_RATE,
        cost_levels=COST_LEVELS,
        severity_profile=PI,
        premium_rate=PREMIUM_RATE,
        n_sim=5000,
        n_periods=10,
        seed=42,
    )
    # Allow some MC variance — at least 3/5 points should satisfy bound
    violations = sum(1 for m, b in zip(psi_mc, psi_bound, strict=True) if m > b * 1.5)
    assert violations <= 2, f"Too many Lundberg violations: {violations}/5"
