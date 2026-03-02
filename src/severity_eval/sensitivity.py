"""Sensitivity analysis: perturbation of costs and severity profiles."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from severity_eval.compound_loss import simulate_aggregate_loss
from severity_eval.risk_measures import compute_risk_measures
from severity_eval.validation import validate_severity_profile


def sensitivity_analysis(
    n_queries: int,
    error_rates: dict[str, float],
    cost_levels: np.ndarray | list[float],
    severity_profiles: dict[str, np.ndarray | list[float]],
    perturbation: float = 0.20,
    n_sim: int = 50000,
    seed: int = 42,
) -> dict:
    """Analyze robustness of model rankings under cost perturbation.

    Perturbs each cost level by ±perturbation and checks whether
    the ranking of models by E[S] remains stable.

    Parameters
    ----------
    n_queries : int
        Queries per period.
    error_rates : dict
        {model_name: error_rate}.
    cost_levels : array-like
        Base cost levels.
    severity_profiles : dict
        {model_name: severity_profile}.
    perturbation : float
        Fraction to perturb costs (e.g. 0.20 for ±20%).
    n_sim : int
        Monte Carlo replications.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys:
        - base_ranking: list of (model, E[S]) sorted by E[S]
        - perturbed_rankings: list of rankings under perturbation
        - spearman_correlations: Spearman ρ between base and each perturbed
        - min_spearman: minimum Spearman ρ (worst case)
    """
    cost_levels = np.asarray(cost_levels, dtype=np.float64)
    for model_name, pi in severity_profiles.items():
        severity_profiles[model_name] = validate_severity_profile(pi)
    models = list(error_rates.keys())

    def compute_ranking(costs: np.ndarray) -> list[tuple[str, float]]:
        results = {}
        for model in models:
            p = error_rates[model]
            pi = np.asarray(severity_profiles[model], dtype=np.float64)
            S = simulate_aggregate_loss(n_queries, p, costs, pi, n_sim, seed)
            measures = compute_risk_measures(S)
            results[model] = measures["expected_loss"]
        ranking = sorted(results.items(), key=lambda x: x[1])
        return ranking

    base_ranking = compute_ranking(cost_levels)
    base_order = [m for m, _ in base_ranking]

    perturbed_rankings = []
    spearman_correlations = []

    K = len(cost_levels)
    for k in range(K):
        for direction in [-1, +1]:
            perturbed_costs = cost_levels.copy()
            perturbed_costs[k] *= 1 + direction * perturbation

            ranking = compute_ranking(perturbed_costs)
            perturbed_order = [m for m, _ in ranking]
            perturbed_rankings.append(
                {
                    "level_perturbed": k,
                    "direction": direction,
                    "costs": perturbed_costs.tolist(),
                    "ranking": ranking,
                }
            )

            # Spearman correlation between base and perturbed rank orders
            base_ranks = np.array([base_order.index(m) for m in models])
            pert_ranks = np.array([perturbed_order.index(m) for m in models])
            if len(models) > 1:
                rho, _ = spearmanr(base_ranks, pert_ranks)
            else:
                rho = 1.0
            spearman_correlations.append(rho)

    return {
        "base_ranking": base_ranking,
        "perturbed_rankings": perturbed_rankings,
        "spearman_correlations": spearman_correlations,
        "min_spearman": min(spearman_correlations) if spearman_correlations else 1.0,
    }
