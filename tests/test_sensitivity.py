"""Tests for severity_eval.sensitivity."""

from __future__ import annotations

import numpy as np
import pytest

from severity_eval.sensitivity import sensitivity_analysis

COSTS = np.array([100.0, 1_000.0, 10_000.0, 100_000.0])
ERROR_RATES = {
    "A": 0.10,
    "B": 0.20,
    "C": 0.05,
}
PROFILES = {
    "A": np.array([0.7, 0.2, 0.1, 0.0]),
    "B": np.array([0.4, 0.3, 0.2, 0.1]),
    "C": np.array([0.1, 0.2, 0.3, 0.4]),
}


def test_returns_expected_keys():
    out = sensitivity_analysis(
        n_queries=1_000,
        error_rates=ERROR_RATES,
        cost_levels=COSTS,
        severity_profiles=PROFILES,
        n_sim=500,
        seed=0,
    )
    assert {
        "base_ranking",
        "perturbed_rankings",
        "spearman_correlations",
        "min_spearman",
    } <= set(out)


def test_base_ranking_is_sorted_by_loss():
    out = sensitivity_analysis(
        n_queries=1_000,
        error_rates=ERROR_RATES,
        cost_levels=COSTS,
        severity_profiles=PROFILES,
        n_sim=500,
        seed=0,
    )
    losses = [loss for _, loss in out["base_ranking"]]
    assert losses == sorted(losses)


def test_perturbed_count_matches_2K_directions():
    """2 directions x K levels of perturbation."""
    out = sensitivity_analysis(
        n_queries=1_000,
        error_rates=ERROR_RATES,
        cost_levels=COSTS,
        severity_profiles=PROFILES,
        n_sim=200,
        seed=0,
    )
    assert len(out["perturbed_rankings"]) == 2 * len(COSTS)
    assert len(out["spearman_correlations"]) == 2 * len(COSTS)


def test_spearman_within_bounds():
    out = sensitivity_analysis(
        n_queries=1_000,
        error_rates=ERROR_RATES,
        cost_levels=COSTS,
        severity_profiles=PROFILES,
        n_sim=500,
        seed=0,
    )
    for rho in out["spearman_correlations"]:
        assert -1.0 <= rho <= 1.0


def test_caller_dict_not_mutated():
    """The function must not modify the caller's severity_profiles dict."""
    profiles = {
        "A": [0.7, 0.2, 0.1, 0.0],  # list, not ndarray
        "B": [0.4, 0.3, 0.2, 0.1],
    }
    snapshot = {k: list(v) for k, v in profiles.items()}
    sensitivity_analysis(
        n_queries=500,
        error_rates={"A": 0.1, "B": 0.2},
        cost_levels=COSTS,
        severity_profiles=profiles,
        n_sim=200,
        seed=0,
    )
    # Original lists must be untouched (no ndarray substitution)
    for k in profiles:
        assert profiles[k] == snapshot[k]
        assert isinstance(profiles[k], list)


def test_perturbation_must_be_in_zero_one():
    for bad in [-0.1, 0.0, 1.0, 1.5]:
        with pytest.raises(ValueError, match="perturbation must be"):
            sensitivity_analysis(
                n_queries=100,
                error_rates=ERROR_RATES,
                cost_levels=COSTS,
                severity_profiles=PROFILES,
                perturbation=bad,
                n_sim=100,
                seed=0,
            )


def test_key_mismatch_raises():
    """error_rates and severity_profiles must share the same model keys."""
    with pytest.raises(ValueError, match="same model keys"):
        sensitivity_analysis(
            n_queries=100,
            error_rates={"A": 0.1},
            cost_levels=COSTS,
            severity_profiles={"B": [0.5, 0.3, 0.1, 0.1]},
            n_sim=100,
            seed=0,
        )


def test_single_model_yields_perfect_rho():
    """With one model, ranking is trivial and rho=1.0."""
    out = sensitivity_analysis(
        n_queries=500,
        error_rates={"only": 0.1},
        cost_levels=COSTS,
        severity_profiles={"only": [0.5, 0.3, 0.1, 0.1]},
        n_sim=200,
        seed=0,
    )
    assert out["min_spearman"] == pytest.approx(1.0)


def test_min_spearman_one_for_well_separated_models():
    """Models whose E[S] differ by orders of magnitude should be
    ordered identically under small cost perturbations."""
    profiles = {
        "tiny": [1.0, 0.0, 0.0, 0.0],
        "big": [0.0, 0.0, 0.0, 1.0],
    }
    error_rates = {"tiny": 0.05, "big": 0.10}
    out = sensitivity_analysis(
        n_queries=1_000,
        error_rates=error_rates,
        cost_levels=COSTS,
        severity_profiles=profiles,
        perturbation=0.2,
        n_sim=500,
        seed=0,
    )
    assert out["min_spearman"] == pytest.approx(1.0)


def test_cost_levels_unchanged_after_call():
    """The cost_levels passed in must not be mutated either."""
    costs = np.array([100.0, 1_000.0, 10_000.0, 100_000.0])
    snapshot = costs.copy()
    sensitivity_analysis(
        n_queries=500,
        error_rates=ERROR_RATES,
        cost_levels=costs,
        severity_profiles=PROFILES,
        n_sim=200,
        seed=0,
    )
    np.testing.assert_array_equal(costs, snapshot)
