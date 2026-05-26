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


def wilson_score_interval(
    n_success: int,
    n_total: int,
    alpha: float = 0.95,
) -> tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion.

    Better-behaved than the normal approximation when the observed
    proportion is near 0 or 1 -- the latter degenerates to a zero-width
    interval there. Use this for reporting accuracy CIs across our
    benchmarks because several model x dataset cells will sit very
    close to the extremes (e.g. gpt-oss-20b on maud at 0.99).

    Parameters
    ----------
    n_success : int
        Number of correct predictions.
    n_total : int
        Total number of predictions. Must be > 0.
    alpha : float
        Confidence level in (0, 1). Default 0.95.

    Returns
    -------
    (lo, hi) : tuple of float
        Lower and upper bounds of the Wilson interval, each in [0, 1].
    """
    if n_total <= 0:
        raise ValueError(f"n_total must be > 0, got {n_total}")
    if not 0 <= n_success <= n_total:
        raise ValueError(f"n_success={n_success} not in [0, n_total={n_total}]")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    # Inverse standard-normal CDF for two-sided alpha. We use the rational
    # approximation rather than depend on scipy for the single quantile.
    # 95% -> z = 1.959963984540054
    half = (1.0 - alpha) / 2.0
    z = float(np.sqrt(2.0) * _erfinv(1.0 - 2.0 * half))

    n = float(n_total)
    p = n_success / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    pad = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, float(centre - pad)), min(1.0, float(centre + pad)))


def _erfinv(x: float) -> float:
    """Inverse error function via the Beasley-Springer-Moro approximation."""
    # Implement here to avoid pulling in scipy.special; accurate to ~6 dp.
    a = 0.147
    log_term = np.log(1.0 - x * x)
    inside = (2.0 / (np.pi * a) + log_term / 2.0) ** 2 - log_term / a
    return float(
        np.sign(x) * np.sqrt(np.sqrt(inside) - (2.0 / (np.pi * a) + log_term / 2.0))
    )


def paired_bootstrap_diff(
    losses_a: np.ndarray,
    losses_b: np.ndarray,
    alpha: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Paired percentile-bootstrap CI for the mean loss difference
    :math:`\\mathbb{E}[L_A] - \\mathbb{E}[L_B]`.

    The pairing is over instances, so :math:`L_A` and :math:`L_B` must be
    aligned per-instance loss vectors (e.g.
    :math:`L_i = \\mathbb{1}\\{\\text{wrong}_i\\} \\cdot c(\\text{severity}_i)`).
    Pairing controls for shared instance difficulty; unpaired bootstrap
    would lose statistical power when the two models err on overlapping
    items.

    Parameters
    ----------
    losses_a, losses_b : ndarray
        Per-instance loss vectors for model A and model B. Must have
        identical length and aligned ordering.
    alpha : float
        Confidence level.
    n_bootstrap : int
        Number of bootstrap resamples.
    seed : int or None
        RNG seed.

    Returns
    -------
    (mean_diff, lo, hi) : tuple of float
        Point estimate of the loss-difference mean and the
        ``alpha``-level CI from the bootstrap distribution.
    """
    if losses_a.shape != losses_b.shape:
        raise ValueError(
            f"losses_a {losses_a.shape} and losses_b {losses_b.shape} must match"
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    rng = np.random.default_rng(seed)
    n = losses_a.size
    diff = losses_a - losses_b
    mean_diff = float(diff.mean())
    stats = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        stats[b] = diff[idx].mean()
    half = (1.0 - alpha) / 2.0
    lo = float(np.quantile(stats, half))
    hi = float(np.quantile(stats, 1.0 - half))
    return (mean_diff, lo, hi)
