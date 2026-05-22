"""Monte Carlo simulation for the compound loss model S = Σ X_i.

Model:
    N ~ Binomial(n, p)       — number of errors per period
    X_i ~ Categorical(c, π)  — severity of each error
    S = Σ_{i=1}^{N} X_i     — aggregate loss
"""

from __future__ import annotations

import numpy as np

from severity_eval.validation import validate_severity_profile


def simulate_aggregate_loss(
    n_queries: int,
    error_rate: float,
    cost_levels: np.ndarray | list[float],
    severity_profile: np.ndarray | list[float],
    n_sim: int = 100000,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate n_sim realisations of the aggregate loss S.

    Parameters
    ----------
    n_queries : int
        Number of queries per period (n in Binomial).
    error_rate : float
        Probability of error per query (p in Binomial).
    cost_levels : array-like
        Dollar cost for each severity level (c_1, ..., c_K).
    severity_profile : array-like
        Probability of each severity level given an error (π_1, ..., π_K).
    n_sim : int
        Number of Monte Carlo replications.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    S : ndarray of shape (n_sim,)
        Simulated aggregate losses.
    """
    cost_levels = np.asarray(cost_levels, dtype=np.float64)
    severity_profile = validate_severity_profile(severity_profile)

    if len(cost_levels) != len(severity_profile):
        raise ValueError("cost_levels and severity_profile must have same length")
    if not (0.0 <= error_rate <= 1.0):
        raise ValueError(f"error_rate must be in [0, 1], got {error_rate}")

    rng = np.random.default_rng(seed)

    # Number of errors per simulation
    N = rng.binomial(n_queries, error_rate, size=n_sim)
    N_total = int(N.sum())

    if N_total == 0:
        return np.zeros(n_sim)

    # For very heavy expected workloads (n_sim * n_queries * p), allocating
    # a flat severities array of length N_total is memory-prohibitive.
    # Chunk over simulations to keep peak memory bounded.
    S = np.zeros(n_sim, dtype=np.float64)

    # Aim for at most ~5e6 categorical draws per chunk (~40MB at float64).
    CHUNK_DRAWS = 5_000_000
    expected_per_sim = max(1, N_total // max(1, n_sim))
    chunk_size = max(1, min(n_sim, CHUNK_DRAWS // expected_per_sim))

    for start in range(0, n_sim, chunk_size):
        end = min(start + chunk_size, n_sim)
        N_chunk = N[start:end]
        total = int(N_chunk.sum())
        if total == 0:
            continue
        severities = rng.choice(cost_levels, size=total, p=severity_profile)
        idx = 0
        for i, ni in enumerate(N_chunk):
            if ni > 0:
                S[start + i] = severities[idx : idx + ni].sum()
                idx += ni

    return S
