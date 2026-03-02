"""Input validation utilities for severity profiles."""

from __future__ import annotations

import numpy as np


def validate_severity_profile(
    severity_profile: np.ndarray | list[float],
    name: str = "severity_profile",
) -> np.ndarray:
    """Validate and normalize a severity profile π.

    Checks:
    1. All values >= 0
    2. If sum ≈ 100, auto-converts from percentages to probabilities
    3. Sum must equal 1 (after potential conversion)

    Parameters
    ----------
    severity_profile : array-like
        Probability (or percentage) vector.
    name : str
        Parameter name for error messages.

    Returns
    -------
    pi : ndarray
        Validated probability vector summing to 1.

    Raises
    ------
    ValueError
        If values are negative or don't sum to 1/100.
    """
    pi = np.asarray(severity_profile, dtype=np.float64)

    if pi.ndim != 1 or len(pi) == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array")

    if (pi < 0).any():
        raise ValueError(f"{name} contains negative values: {pi[pi < 0].tolist()}")

    total = pi.sum()

    # Auto-detect percentages: sum ≈ 100
    if np.isclose(total, 100.0, atol=1.0) and not np.isclose(total, 1.0):
        pi = pi / 100.0
        total = pi.sum()

    if not np.isclose(total, 1.0):
        raise ValueError(f"{name} must sum to 1 (or 100 for percentages), got {total:.6f}")

    return pi
