"""HITL routing analysis: retention threshold, cost reduction, and routing ratio."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from severity_eval.validation import validate_severity_profile


@dataclass
class RoutingResult:
    """Results of a HITL routing analysis.

    Attributes — input decomposition
    ---------------------------------
    cost_levels : original c_k vector
    severity_profile : original π_k vector
    retained_cost_levels : c_k for levels below threshold
    retained_severity_profile : renormalized π_k^ret (sums to 1)
    routed_cost_levels : c_k for levels at or above threshold
    routed_severity_profile : renormalized π_k^route (sums to 1)

    Attributes — scalar measures
    ----------------------------
    mu_X : E[X] = Σ π_k c_k  (mean severity, unrouted)
    mu_X_retained : E[X_ret]  (mean severity, retained only)
    """

    # Configuration
    retention_threshold: int | float
    human_review_cost: int | float

    # Full input vectors
    cost_levels: np.ndarray
    severity_profile: np.ndarray

    # Decomposition — retained (AI handles)
    retained_cost_levels: np.ndarray
    retained_severity_profile: np.ndarray

    # Decomposition — routed (human handles)
    routed_cost_levels: np.ndarray
    routed_severity_profile: np.ndarray

    # Mean severities
    mu_X: float  # E[X] overall
    mu_X_retained: float  # E[X_ret] for retained

    # Volumes
    routing_ratio: float  # ρ ∈ [0, 1]
    n_routed: int
    n_retained: int

    # Dollar measures
    expected_loss_unrouted: float  # E[S] = n·p·μ_X
    expected_loss_retained: float  # E[S_ret] = n_ret·p·μ_X_ret
    routing_cost: float  # n_routed · h
    expected_total_cost: float  # E[C] = E[S_ret] + routing_cost
    cost_reduction_pct: float  # (1 − E[C]/E[S]) × 100


def analyze_routing(
    n_queries: int,
    error_rate: float,
    cost_levels: np.ndarray | list[int] | list[float],
    severity_profile: np.ndarray | list[float],
    retention_threshold: int | float,
    human_review_cost: int | float = 50,
) -> RoutingResult:
    """Analyze the impact of HITL routing with a retention threshold.

    Errors with severity cost ≥ retention_threshold are routed to human
    review at a fixed cost h per query. Retained errors are absorbed
    by the AI system at their actuarial cost.

    Parameters
    ----------
    n_queries : int
        Total number of queries per period.
    error_rate : float
        Error probability per query (p).
    cost_levels : array-like of int or float
        Dollar cost for each severity level (c_1, …, c_K).
    severity_profile : array-like
        Probability of each severity level given error (π_1, …, π_K).
    retention_threshold : int or float
        Cost threshold d: levels with c_k ≥ d are routed to humans.
    human_review_cost : int or float
        Cost per routed query for human review (h).

    Returns
    -------
    RoutingResult
        Full breakdown with π_k, c_k for retained and routed partitions.
    """
    cost_levels = np.asarray(cost_levels, dtype=np.float64)
    severity_profile = validate_severity_profile(severity_profile)

    # Partition levels into retained (AI) vs routed (human)
    retained_mask = cost_levels < retention_threshold
    routed_mask = ~retained_mask

    c_retained = cost_levels[retained_mask]
    c_routed = cost_levels[routed_mask]
    pi_retained_raw = severity_profile[retained_mask]
    pi_routed_raw = severity_profile[routed_mask]

    # Routing ratio: fraction of error-queries that get routed
    rho = float(pi_routed_raw.sum())

    # Renormalized profiles
    pi_ret_sum = pi_retained_raw.sum()
    if pi_ret_sum > 0:
        pi_retained = pi_retained_raw / pi_ret_sum
        mu_X_retained = float((c_retained * pi_retained).sum())
    else:
        pi_retained = np.zeros_like(pi_retained_raw)
        mu_X_retained = 0.0

    pi_rout_sum = pi_routed_raw.sum()
    if pi_rout_sum > 0:
        pi_routed = pi_routed_raw / pi_rout_sum
    else:
        pi_routed = np.zeros_like(pi_routed_raw)

    # Mean severity (overall, unrouted)
    mu_X = float((cost_levels * severity_profile).sum())

    # Volume split
    n_routed = int(n_queries * rho)
    n_retained = n_queries - n_routed

    # Dollar measures
    expected_loss_unrouted = n_queries * error_rate * mu_X
    expected_loss_retained = n_retained * error_rate * mu_X_retained
    routing_cost = n_routed * error_rate * float(human_review_cost)
    expected_total_cost = expected_loss_retained + routing_cost

    if expected_loss_unrouted > 0:
        cost_reduction_pct = (1 - expected_total_cost / expected_loss_unrouted) * 100
    else:
        cost_reduction_pct = 0.0

    return RoutingResult(
        retention_threshold=retention_threshold,
        human_review_cost=human_review_cost,
        cost_levels=cost_levels,
        severity_profile=severity_profile,
        retained_cost_levels=c_retained,
        retained_severity_profile=pi_retained,
        routed_cost_levels=c_routed,
        routed_severity_profile=pi_routed,
        mu_X=mu_X,
        mu_X_retained=mu_X_retained,
        routing_ratio=rho,
        n_routed=n_routed,
        n_retained=n_retained,
        expected_loss_unrouted=expected_loss_unrouted,
        expected_loss_retained=expected_loss_retained,
        routing_cost=routing_cost,
        expected_total_cost=expected_total_cost,
        cost_reduction_pct=cost_reduction_pct,
    )
