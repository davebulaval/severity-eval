"""Tests for sensitivity module."""

import numpy as np

from severity_eval.sensitivity import sensitivity_analysis

COST_LEVELS = np.array([100, 1000, 10000, 100000])


def test_sensitivity_stable_under_small_perturbation():
    """Rankings should be stable under ±20% cost perturbation when models are well-separated."""
    error_rates = {
        "model_A": 0.05,
        "model_B": 0.10,
        "model_C": 0.15,
    }
    severity_profiles = {
        "model_A": np.array([0.4, 0.3, 0.2, 0.1]),
        "model_B": np.array([0.3, 0.3, 0.2, 0.2]),
        "model_C": np.array([0.2, 0.2, 0.3, 0.3]),
    }
    result = sensitivity_analysis(
        n_queries=10000,
        error_rates=error_rates,
        cost_levels=COST_LEVELS,
        severity_profiles=severity_profiles,
        perturbation=0.20,
        n_sim=50000,
        seed=42,
    )
    assert result["min_spearman"] >= 0.5, f"Rankings too unstable: min ρ = {result['min_spearman']}"


def test_sensitivity_output_structure():
    """Output should have expected keys."""
    error_rates = {"A": 0.05, "B": 0.10}
    severity_profiles = {
        "A": np.array([0.5, 0.3, 0.15, 0.05]),
        "B": np.array([0.3, 0.3, 0.2, 0.2]),
    }
    result = sensitivity_analysis(
        n_queries=10000,
        error_rates=error_rates,
        cost_levels=COST_LEVELS,
        severity_profiles=severity_profiles,
        n_sim=10000,
        seed=42,
    )
    assert "base_ranking" in result
    assert "perturbed_rankings" in result
    assert "spearman_correlations" in result
    assert "min_spearman" in result
    # 4 cost levels × 2 directions = 8 perturbations
    assert len(result["perturbed_rankings"]) == 8
