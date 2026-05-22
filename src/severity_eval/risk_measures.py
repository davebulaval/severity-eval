"""Risk measures: expected loss, VaR, TVaR, and bootstrap confidence intervals.

Empirical estimators that operate on a Monte Carlo sample of aggregate
losses ``S`` produced by :func:`severity_eval.compound_loss.simulate_aggregate_loss`.
"""

from __future__ import annotations

import numpy as np


def compute_risk_measures(
    losses: np.ndarray,
    alpha: float = 0.95,
) -> dict[str, float]:
    """Compute expected loss, VaR and TVaR from a Monte Carlo sample.

    Parameters
    ----------
    losses : ndarray
        Aggregate loss realisations ``S_1, ..., S_M``.
    alpha : float
        Confidence level in (0, 1) for VaR / TVaR. Default 0.95.

    Returns
    -------
    dict with keys ``expected_loss``, ``var``, ``tvar``.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    losses = np.asarray(losses, dtype=np.float64)
    if losses.size == 0:
        return {"expected_loss": 0.0, "var": 0.0, "tvar": 0.0}

    expected_loss = float(losses.mean())
    var = float(np.quantile(losses, alpha))

    tail = losses[losses > var]
    if tail.size == 0:
        # Degenerate: when many losses tie at the VaR (e.g. zero-loss mass)
        # use the upper tail including the quantile point.
        tail = losses[losses >= var]
    tvar = float(tail.mean()) if tail.size > 0 else var

    return {"expected_loss": expected_loss, "var": var, "tvar": tvar}


def bootstrap_ci(
    losses: np.ndarray,
    statistic: str = "expected_loss",
    alpha: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a loss statistic.

    Parameters
    ----------
    losses : ndarray
        Monte Carlo aggregate losses.
    statistic : {'expected_loss', 'var', 'tvar'}
        Which functional of the loss distribution to bootstrap.
    alpha : float
        Confidence level (default 0.95 → 2.5% / 97.5% percentiles).
    n_bootstrap : int
        Number of bootstrap resamples.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    (lo, hi) : tuple of float
        Lower and upper bounds of the percentile CI.
    """
    if statistic not in ("expected_loss", "var", "tvar"):
        raise ValueError(f"Unknown statistic {statistic!r}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")
    losses = np.asarray(losses, dtype=np.float64)
    if losses.size == 0:
        return (0.0, 0.0)

    rng = np.random.default_rng(seed)
    n = losses.size
    stats = np.empty(n_bootstrap, dtype=np.float64)

    for b in range(n_bootstrap):
        sample = losses[rng.integers(0, n, size=n)]
        if statistic == "expected_loss":
            stats[b] = sample.mean()
        elif statistic == "var":
            stats[b] = np.quantile(sample, alpha)
        else:  # tvar
            q = np.quantile(sample, alpha)
            tail = sample[sample > q]
            stats[b] = tail.mean() if tail.size > 0 else q

    half = (1.0 - alpha) / 2.0
    lo = float(np.quantile(stats, half))
    hi = float(np.quantile(stats, 1.0 - half))
    return (lo, hi)
