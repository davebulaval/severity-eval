"""Risk measures: E[S], VaR, TVaR with bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np


def compute_risk_measures(
    S: np.ndarray,
    alpha: float = 0.95,
) -> dict[str, float]:
    """Compute expected loss, VaR, and TVaR from simulated losses.

    Parameters
    ----------
    S : ndarray
        Simulated aggregate losses.
    alpha : float
        Confidence level for VaR/TVaR (e.g. 0.95).

    Returns
    -------
    dict with keys: expected_loss, var, tvar.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    expected_loss = float(S.mean())
    var = float(np.quantile(S, alpha))
    # Formula-based TVaR: robust for both continuous and discrete distributions
    # TVaR_α = VaR_α + E[max(S - VaR_α, 0)] / (1 - α)
    tvar = var + float(np.mean(np.maximum(S - var, 0))) / (1 - alpha)

    return {
        "expected_loss": expected_loss,
        "var": var,
        "tvar": tvar,
    }


def bootstrap_ci(
    S: np.ndarray,
    statistic: str = "expected_loss",
    alpha: float = 0.95,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for a risk measure.

    Parameters
    ----------
    S : ndarray
        Simulated aggregate losses.
    statistic : str
        One of 'expected_loss', 'var', 'tvar'.
    alpha : float
        Confidence level for VaR/TVaR.
    confidence : float
        CI confidence level (e.g. 0.95 for 95% CI).
    n_bootstrap : int
        Number of bootstrap samples.
    seed : int or None
        Random seed.

    Returns
    -------
    (lower, upper) : tuple of float
    """
    rng = np.random.default_rng(seed)
    n = len(S)
    estimates = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        sample = S[rng.integers(0, n, size=n)]
        measures = compute_risk_measures(sample, alpha=alpha)
        estimates[b] = measures[statistic]

    lo = float(np.percentile(estimates, (1 - confidence) / 2 * 100))
    hi = float(np.percentile(estimates, (1 + confidence) / 2 * 100))
    return lo, hi
