"""Ruin theory: Lundberg adjustment coefficient and minimum reserve."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from severity_eval.validation import validate_severity_profile


def compute_lundberg_R(
    claim_rate: float,
    cost_levels: np.ndarray | list[float],
    severity_profile: np.ndarray | list[float],
    premium_rate: float,
) -> float:
    """Compute the Lundberg adjustment coefficient R.

    Solves: λ(M_X(r) - 1) = c·r
    where M_X(r) = Σ_k π_k exp(r·c_k) is the moment generating function.

    Parameters
    ----------
    claim_rate : float
        Expected number of claims per period (λ = n·p).
    cost_levels : array-like
        Dollar cost for each severity level.
    severity_profile : array-like
        Probability of each severity level.
    premium_rate : float
        Premium income per period.

    Returns
    -------
    R : float
        The adjustment coefficient (positive root).
    """
    cost_levels = np.asarray(cost_levels, dtype=np.float64)
    severity_profile = validate_severity_profile(severity_profile)

    def equation(r: float) -> float:
        mx = np.sum(severity_profile * np.exp(r * cost_levels))
        return claim_rate * (mx - 1) - premium_rate * r

    # Search for the positive root in expanding intervals
    for upper in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        try:
            R = brentq(equation, 1e-15, upper, xtol=1e-15)
            return float(R)
        except ValueError:
            continue

    raise RuntimeError("Could not find Lundberg adjustment coefficient")


def compute_reserve(R: float, ruin_target: float = 0.01) -> float:
    """Minimum reserve from Lundberg bound: ψ(u) ≤ exp(-Ru).

    Parameters
    ----------
    R : float
        Adjustment coefficient.
    ruin_target : float
        Target ruin probability (e.g. 0.01 for 1%).

    Returns
    -------
    u_star : float
        Minimum initial reserve.
    """
    if R <= 0:
        raise ValueError(f"R must be positive, got {R}")
    if not (0 < ruin_target < 1):
        raise ValueError(f"ruin_target must be in (0, 1), got {ruin_target}")

    return float(np.log(1.0 / ruin_target) / R)


def simulate_ruin_probability(
    initial_reserve: int | float,
    claim_rate: float,
    cost_levels: np.ndarray | list[float],
    severity_profile: np.ndarray | list[float],
    premium_rate: float,
    n_sim: int = 10000,
    n_periods: int = 10,
    seed: int | None = None,
) -> float:
    """Estimate ruin probability ψ(u) via Monte Carlo.

    Simulates the surplus process:
        U(t) = u + c·t - S(t)
    where S(t) is the aggregate claim process (compound Poisson approximated
    by compound Binomial).

    Ruin occurs if U(t) < 0 for any t in [1, n_periods].

    Parameters
    ----------
    initial_reserve : int or float
        Initial surplus u.
    claim_rate : float
        Expected claims per period (λ = n·p).
    cost_levels : array-like
        Dollar cost for each severity level.
    severity_profile : array-like
        Probability of each severity level.
    premium_rate : float
        Premium income per period (c).
    n_sim : int
        Number of Monte Carlo paths.
    n_periods : int
        Number of time periods to simulate.
    seed : int or None
        Random seed.

    Returns
    -------
    psi : float
        Estimated ruin probability ψ(u).
    """
    cost_levels = np.asarray(cost_levels, dtype=np.float64)
    severity_profile = validate_severity_profile(severity_profile)
    rng = np.random.default_rng(seed)

    ruin_count = 0

    for _ in range(n_sim):
        U = float(initial_reserve)
        ruined = False
        for _ in range(n_periods):
            N_t = rng.poisson(claim_rate)
            if N_t > 0:
                X_t = rng.choice(cost_levels, size=N_t, p=severity_profile).sum()
            else:
                X_t = 0.0
            U = U + premium_rate - X_t
            if U < 0:
                ruined = True
                break
        if ruined:
            ruin_count += 1

    return ruin_count / n_sim


def ruin_probability_curve(
    u_range: np.ndarray | list[float],
    claim_rate: float,
    cost_levels: np.ndarray | list[float],
    severity_profile: np.ndarray | list[float],
    premium_rate: float,
    n_sim: int = 10000,
    n_periods: int = 10,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ψ_MC(u) and Lundberg bound exp(-Ru) over a grid of u values.

    Parameters
    ----------
    u_range : array-like
        Grid of initial reserve values.
    claim_rate, cost_levels, severity_profile, premium_rate :
        Model parameters (see simulate_ruin_probability).
    n_sim, n_periods, seed :
        MC parameters.

    Returns
    -------
    u_values : ndarray
        The reserve grid.
    psi_mc : ndarray
        MC ruin probability estimates.
    psi_bound : ndarray
        Lundberg upper bound exp(-R·u).
    """
    u_values = np.asarray(u_range, dtype=np.float64)
    cost_levels = np.asarray(cost_levels, dtype=np.float64)
    severity_profile = validate_severity_profile(severity_profile)

    R = compute_lundberg_R(claim_rate, cost_levels, severity_profile, premium_rate)
    psi_bound = np.exp(-R * u_values)

    psi_mc = np.empty(len(u_values))
    for i, u in enumerate(u_values):
        psi_mc[i] = simulate_ruin_probability(
            initial_reserve=u,
            claim_rate=claim_rate,
            cost_levels=cost_levels,
            severity_profile=severity_profile,
            premium_rate=premium_rate,
            n_sim=n_sim,
            n_periods=n_periods,
            seed=seed + i if seed is not None else None,
        )

    return u_values, psi_mc, psi_bound
