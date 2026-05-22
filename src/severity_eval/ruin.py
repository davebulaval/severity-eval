"""Ruin theory: Lundberg adjustment coefficient and reserve requirements.

Implements the Cramer-Lundberg model under a compound Poisson approximation
of the aggregate loss process:

    U_t = u + c t - sum_{k=1}^{N(t)} X_k,   N(t) ~ Poisson(lambda t).

The adjustment coefficient ``R`` is the positive root of

    lambda (M_X(R) - 1) = c R,

where ``M_X`` is the MGF of the categorical severity distribution and
``c`` is the premium rate. Lundberg's inequality gives
``psi(u) <= exp(-R u)``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from severity_eval.validation import validate_severity_profile


def _mgf_minus_1(
    r: float,
    cost_levels: np.ndarray,
    severity_profile: np.ndarray,
) -> float:
    """Return ``M_X(r) - 1`` for a categorical severity distribution."""
    # exp(r * c_k) - 1 is more numerically stable than computing M_X then
    # subtracting 1, especially near r = 0.
    return float(np.sum(severity_profile * np.expm1(r * cost_levels)))


def compute_lundberg_R(
    claim_rate: float,
    cost_levels: np.ndarray | list[float],
    severity_profile: np.ndarray | list[float],
    premium_rate: float,
    r_max: float = 1e-3,
) -> float:
    """Solve for the Lundberg adjustment coefficient R > 0.

    R is the unique positive root of ``lambda (M_X(R) - 1) = c R``,
    which exists iff ``c > lambda * E[X]`` (positive safety loading).

    Parameters
    ----------
    claim_rate : float
        Poisson intensity ``lambda = n * p``.
    cost_levels : array-like
        Severity costs ``c_1, ..., c_K``.
    severity_profile : array-like
        Probabilities ``pi_1, ..., pi_K`` summing to 1.
    premium_rate : float
        Premium income per unit time, ``c``.
    r_max : float
        Upper bracket for the root search; defaults to ``1e-3`` which
        comfortably covers the realistic range for the cost vectors used
        in this paper. Reduce automatically if the bracket diverges.

    Returns
    -------
    R : float
        Positive adjustment coefficient.

    Raises
    ------
    ValueError
        If the safety loading is non-positive (no positive root exists).
    """
    cost_levels = np.asarray(cost_levels, dtype=np.float64)
    severity_profile = validate_severity_profile(severity_profile)
    if cost_levels.shape != severity_profile.shape:
        raise ValueError("cost_levels and severity_profile must have same shape")

    mu_X = float((cost_levels * severity_profile).sum())
    expected_loss_rate = claim_rate * mu_X
    if premium_rate <= expected_loss_rate:
        raise ValueError(
            "Premium rate must exceed expected loss rate "
            f"(got premium={premium_rate}, lambda*E[X]={expected_loss_rate})"
        )

    def f(r: float) -> float:
        return (
            claim_rate * _mgf_minus_1(r, cost_levels, severity_profile)
            - premium_rate * r
        )

    # Find an upper bracket where f changes sign. The MGF grows
    # exponentially in r; for heavy tails r_max may need to shrink.
    upper = r_max
    f_upper = f(upper)
    while not np.isfinite(f_upper) or f_upper <= 0:
        upper /= 10.0
        if upper < 1e-15:
            raise RuntimeError("Could not find an upper bracket for R")
        f_upper = f(upper)

    # f(0) = 0; perturb slightly to get a strictly negative lower bracket.
    lower = upper * 1e-6
    f_lower = f(lower)
    while f_lower >= 0:
        lower /= 10.0
        if lower < 1e-30:
            raise RuntimeError("Could not find a lower bracket for R")
        f_lower = f(lower)

    return float(brentq(f, lower, upper, xtol=1e-12, rtol=1e-10))


def compute_reserve(R: float, ruin_target: float = 0.01) -> float:
    """Lundberg-bound reserve ``u* = -ln(psi*) / R``.

    Parameters
    ----------
    R : float
        Adjustment coefficient (must be strictly positive).
    ruin_target : float
        Maximum acceptable ruin probability ``psi*``, in (0, 1).

    Returns
    -------
    u_star : float
        Smallest reserve such that the Lundberg bound ``exp(-R u)``
        does not exceed ``ruin_target``.
    """
    if R <= 0:
        raise ValueError(f"R must be positive, got {R}")
    if not (0.0 < ruin_target < 1.0):
        raise ValueError(f"ruin_target must be in (0, 1), got {ruin_target}")
    return float(-np.log(ruin_target) / R)


def simulate_ruin_probability(
    initial_reserve: float,
    claim_rate: float,
    cost_levels: np.ndarray | list[float],
    severity_profile: np.ndarray | list[float],
    premium_rate: float,
    n_sim: int = 10_000,
    n_periods: int = 10,
    seed: int | None = None,
) -> float:
    """Monte Carlo estimate of the finite-horizon ruin probability ``psi(u)``.

    Each period ``t`` adds ``premium_rate`` to the surplus and subtracts
    ``sum X_k`` for ``N_t ~ Poisson(claim_rate)`` independent severity
    draws. Ruin occurs the first time ``U_t < 0`` within ``n_periods``
    periods.

    Returns
    -------
    psi : float
        Empirical ruin probability in [0, 1].
    """
    cost_levels = np.asarray(cost_levels, dtype=np.float64)
    severity_profile = validate_severity_profile(severity_profile)

    rng = np.random.default_rng(seed)

    # Pre-draw all claim counts for vectorisation.
    counts = rng.poisson(claim_rate, size=(n_sim, n_periods))
    ruin = np.zeros(n_sim, dtype=bool)

    for i in range(n_sim):
        if ruin[i]:
            continue
        u = initial_reserve
        for t in range(n_periods):
            n_t = counts[i, t]
            if n_t > 0:
                claims = rng.choice(
                    cost_levels, size=int(n_t), p=severity_profile
                ).sum()
            else:
                claims = 0.0
            u = u + premium_rate - claims
            if u < 0:
                ruin[i] = True
                break
    return float(ruin.mean())


def ruin_probability_curve(
    u_range: np.ndarray | list[float],
    claim_rate: float,
    cost_levels: np.ndarray | list[float],
    severity_profile: np.ndarray | list[float],
    premium_rate: float,
    n_sim: int = 5_000,
    n_periods: int = 10,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Monte Carlo and Lundberg ruin probabilities over a reserve grid.

    Returns
    -------
    u_values : ndarray
        The input reserve grid.
    psi_mc : ndarray
        Empirical ruin probabilities estimated by Monte Carlo.
    psi_bound : ndarray
        Lundberg upper bound ``exp(-R u)``.
    """
    u_values = np.asarray(u_range, dtype=np.float64)
    R = compute_lundberg_R(claim_rate, cost_levels, severity_profile, premium_rate)
    psi_bound = np.exp(-R * u_values)

    psi_mc = np.zeros_like(u_values)
    for i, u in enumerate(u_values):
        psi_mc[i] = simulate_ruin_probability(
            initial_reserve=float(u),
            claim_rate=claim_rate,
            cost_levels=cost_levels,
            severity_profile=severity_profile,
            premium_rate=premium_rate,
            n_sim=n_sim,
            n_periods=n_periods,
            seed=seed,
        )
    return u_values, psi_mc, psi_bound
