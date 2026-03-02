"""Tests for routing module (T14-T16)."""

import numpy as np
import pytest

from severity_eval.routing import analyze_routing

COST_LEVELS = np.array([100, 1000, 10000, 100000])
PI = np.array([0.4, 0.3, 0.2, 0.1])
N_QUERIES = 10000
P = 0.05
HUMAN_COST = 50


# T14: E[C] < E[S] for reasonable d
def test_routing_reduces_expected_cost():
    """Routing with reasonable d should reduce total cost."""
    result = analyze_routing(
        n_queries=N_QUERIES,
        error_rate=P,
        cost_levels=COST_LEVELS,
        severity_profile=PI,
        retention_threshold=10000,
        human_review_cost=HUMAN_COST,
    )
    assert result.expected_total_cost < result.expected_loss_unrouted
    assert result.cost_reduction_pct > 0


# T15: ρ=0 when d ≥ c_max
def test_no_routing_when_threshold_high():
    """No queries routed when d > max cost level."""
    result = analyze_routing(
        n_queries=N_QUERIES,
        error_rate=P,
        cost_levels=COST_LEVELS,
        severity_profile=PI,
        retention_threshold=200000,
        human_review_cost=HUMAN_COST,
    )
    assert result.routing_ratio == 0.0
    assert result.n_routed == 0
    assert result.expected_total_cost == pytest.approx(result.expected_loss_unrouted, rel=1e-6)


# T16: ρ≈1 when d < c_min
def test_all_routed_when_threshold_low():
    """All queries routed when d < min cost level."""
    result = analyze_routing(
        n_queries=N_QUERIES,
        error_rate=P,
        cost_levels=COST_LEVELS,
        severity_profile=PI,
        retention_threshold=50,
        human_review_cost=HUMAN_COST,
    )
    assert result.routing_ratio == pytest.approx(1.0)
    assert result.n_routed >= N_QUERIES - 1


def test_routing_ratio_intermediate():
    """Routing ratio should match probability mass above threshold."""
    result = analyze_routing(
        n_queries=N_QUERIES,
        error_rate=P,
        cost_levels=COST_LEVELS,
        severity_profile=PI,
        retention_threshold=10000,
        human_review_cost=HUMAN_COST,
    )
    expected_rho = PI[2] + PI[3]  # 0.2 + 0.1 = 0.3
    assert result.routing_ratio == pytest.approx(expected_rho)


# ------------------------------------------------------------------
# NEW: π_k and c_k decomposition
# ------------------------------------------------------------------


def test_retained_cost_levels():
    """Retained cost levels are those strictly below threshold."""
    result = analyze_routing(
        n_queries=N_QUERIES,
        error_rate=P,
        cost_levels=COST_LEVELS,
        severity_profile=PI,
        retention_threshold=10000,
        human_review_cost=HUMAN_COST,
    )
    np.testing.assert_array_equal(result.retained_cost_levels, [100, 1000])
    np.testing.assert_array_equal(result.routed_cost_levels, [10000, 100000])


def test_retained_severity_profile_sums_to_one():
    """Retained π_k^ret should sum to 1."""
    result = analyze_routing(
        n_queries=N_QUERIES,
        error_rate=P,
        cost_levels=COST_LEVELS,
        severity_profile=PI,
        retention_threshold=10000,
        human_review_cost=HUMAN_COST,
    )
    assert result.retained_severity_profile.sum() == pytest.approx(1.0)
    # π_ret = [0.4, 0.3] / 0.7 ≈ [0.571, 0.429]
    assert result.retained_severity_profile[0] == pytest.approx(0.4 / 0.7)
    assert result.retained_severity_profile[1] == pytest.approx(0.3 / 0.7)


def test_routed_severity_profile_sums_to_one():
    """Routed π_k^route should sum to 1."""
    result = analyze_routing(
        n_queries=N_QUERIES,
        error_rate=P,
        cost_levels=COST_LEVELS,
        severity_profile=PI,
        retention_threshold=10000,
        human_review_cost=HUMAN_COST,
    )
    assert result.routed_severity_profile.sum() == pytest.approx(1.0)
    # π_route = [0.2, 0.1] / 0.3 ≈ [0.667, 0.333]
    assert result.routed_severity_profile[0] == pytest.approx(0.2 / 0.3)
    assert result.routed_severity_profile[1] == pytest.approx(0.1 / 0.3)


def test_mu_X_values():
    """μ_X and μ_X_retained should be correct."""
    result = analyze_routing(
        n_queries=N_QUERIES,
        error_rate=P,
        cost_levels=COST_LEVELS,
        severity_profile=PI,
        retention_threshold=10000,
        human_review_cost=HUMAN_COST,
    )
    # μ_X = 0.4×100 + 0.3×1000 + 0.2×10000 + 0.1×100000 = 12340
    assert result.mu_X == pytest.approx(12340.0)
    # μ_X_ret = (0.4/0.7)×100 + (0.3/0.7)×1000 ≈ 485.71
    assert result.mu_X_retained == pytest.approx(100 * 0.4 / 0.7 + 1000 * 0.3 / 0.7)


def test_original_vectors_preserved():
    """The full cost_levels and severity_profile should be stored."""
    result = analyze_routing(
        n_queries=N_QUERIES,
        error_rate=P,
        cost_levels=COST_LEVELS,
        severity_profile=PI,
        retention_threshold=10000,
        human_review_cost=HUMAN_COST,
    )
    np.testing.assert_array_equal(result.cost_levels, COST_LEVELS)
    np.testing.assert_array_equal(result.severity_profile, PI)


def test_accepts_int_threshold():
    """Threshold and human cost should accept int."""
    result = analyze_routing(
        n_queries=N_QUERIES,
        error_rate=P,
        cost_levels=[100, 1000, 10000, 100000],
        severity_profile=PI,
        retention_threshold=10000,
        human_review_cost=50,
    )
    assert result.retention_threshold == 10000
    assert result.human_review_cost == 50
    assert result.expected_total_cost < result.expected_loss_unrouted


def test_no_retained_levels():
    """When all levels are routed, retained profile is empty/zero."""
    result = analyze_routing(
        n_queries=N_QUERIES,
        error_rate=P,
        cost_levels=COST_LEVELS,
        severity_profile=PI,
        retention_threshold=50,
        human_review_cost=HUMAN_COST,
    )
    assert len(result.retained_cost_levels) == 0
    assert result.mu_X_retained == 0.0
