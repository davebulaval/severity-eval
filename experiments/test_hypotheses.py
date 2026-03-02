"""Test all 5 hypotheses from the paper plan.

Usage:
    python experiments/test_hypotheses.py --results-dir results/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from severity_eval.compound_loss import simulate_aggregate_loss
from severity_eval.routing import analyze_routing
from severity_eval.sensitivity import sensitivity_analysis
from severity_eval.taxonomy import get_taxonomy


def test_h1(metrics_df: pd.DataFrame) -> dict:
    """H1 — Divergence de classement.

    Le classement par E[S] diverge significativement du classement par accuracy
    (τ Kendall < 0.5) sur au moins 3 domaines.
    """
    results = {}
    divergent_count = 0

    for domain in metrics_df["domain"].unique():
        df = metrics_df[metrics_df["domain"] == domain]
        if len(df) < 3:
            continue

        acc_rank = df["accuracy"].rank(ascending=False)
        loss_rank = df["expected_loss"].rank(ascending=True)
        tau, p_value = kendalltau(acc_rank, loss_rank)

        is_divergent = tau < 0.5 and p_value < 0.05
        if is_divergent:
            divergent_count += 1

        results[domain] = {"tau": tau, "p_value": p_value, "divergent": is_divergent}

    return {
        "per_domain": results,
        "n_divergent": divergent_count,
        "hypothesis_supported": divergent_count >= 3,
    }


def test_h2(metrics_df: pd.DataFrame, n_queries: int = 10_000, n_sim: int = 50_000) -> dict:
    """H2 — Dominance de la sévérité sur la fréquence.

    La variance inter-modèles de E[S] est davantage expliquée par les différences
    de profil de sévérité (π) que par les différences de taux d'erreur (p).
    """
    results = {}

    for domain in metrics_df["domain"].unique():
        df = metrics_df[metrics_df["domain"] == domain]
        if len(df) < 3:
            continue
        taxonomy = get_taxonomy(domain)

        # Baseline E[S] values
        baseline_es = df["expected_loss"].values

        # Fix error rate to mean, vary π → E[S]_π
        mean_p = df["error_rate"].mean()
        es_fixed_p = []
        for _, row in df.iterrows():
            pi = np.array(row["severity_profile"])
            if pi.sum() == 0:
                es_fixed_p.append(0.0)
                continue
            S = simulate_aggregate_loss(n_queries, mean_p, taxonomy.cost_levels, pi, n_sim, seed=42)
            es_fixed_p.append(S.mean())

        # Fix π to mean, vary p → E[S]_p
        profiles = [np.array(row["severity_profile"]) for _, row in df.iterrows()]
        mean_pi = np.mean(profiles, axis=0)
        if mean_pi.sum() > 0:
            mean_pi = mean_pi / mean_pi.sum()

        es_fixed_pi = []
        for _, row in df.iterrows():
            if mean_pi.sum() == 0:
                es_fixed_pi.append(0.0)
                continue
            S = simulate_aggregate_loss(n_queries, row["error_rate"], taxonomy.cost_levels, mean_pi, n_sim, seed=42)
            es_fixed_pi.append(S.mean())

        var_baseline = np.var(baseline_es)
        var_severity = np.var(es_fixed_p)  # only π varies
        var_frequency = np.var(es_fixed_pi)  # only p varies

        results[domain] = {
            "var_baseline": var_baseline,
            "var_severity_only": var_severity,
            "var_frequency_only": var_frequency,
            "severity_dominates": var_severity > var_frequency,
        }

    n_severity_dominates = sum(1 for v in results.values() if v["severity_dominates"])
    return {
        "per_domain": results,
        "hypothesis_supported": n_severity_dominates > len(results) / 2,
    }


def test_h3(metrics_df: pd.DataFrame) -> dict:
    """H3 — Inversions de classement.

    Il existe des paires (A, B) où accuracy(A) > accuracy(B) mais E[S](A) > E[S](B).
    """
    inversions = []

    for domain in metrics_df["domain"].unique():
        df = metrics_df[metrics_df["domain"] == domain]
        models = df.to_dict("records")

        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a, b = models[i], models[j]
                # A has higher accuracy but also higher expected loss
                if a["accuracy"] > b["accuracy"] and a["expected_loss"] > b["expected_loss"]:
                    inversions.append(
                        {
                            "domain": domain,
                            "better_acc": a["model"],
                            "worse_acc": b["model"],
                            "acc_gap": a["accuracy"] - b["accuracy"],
                            "loss_gap_dollars": a["expected_loss"] - b["expected_loss"],
                        }
                    )
                elif b["accuracy"] > a["accuracy"] and b["expected_loss"] > a["expected_loss"]:
                    inversions.append(
                        {
                            "domain": domain,
                            "better_acc": b["model"],
                            "worse_acc": a["model"],
                            "acc_gap": b["accuracy"] - a["accuracy"],
                            "loss_gap_dollars": b["expected_loss"] - a["expected_loss"],
                        }
                    )

    return {
        "n_inversions": len(inversions),
        "inversions": inversions,
        "hypothesis_supported": len(inversions) > 0,
    }


def test_h4(metrics_df: pd.DataFrame, n_queries: int = 10_000) -> dict:
    """H4 — Bénéfice du routage proportionnel à la lourdeur de queue.

    Le % de réduction par routage est plus élevé pour les domaines à queue lourde.
    """
    domain_results = {}

    for domain in metrics_df["domain"].unique():
        taxonomy = get_taxonomy(domain)
        df = metrics_df[metrics_df["domain"] == domain]

        reductions = []
        for _, row in df.iterrows():
            pi = np.array(row["severity_profile"])
            if row["error_rate"] == 0 or pi.sum() == 0:
                continue
            result = analyze_routing(
                n_queries=n_queries,
                error_rate=row["error_rate"],
                cost_levels=taxonomy.cost_levels,
                severity_profile=pi,
                retention_threshold=taxonomy.cost_levels[2],
            )
            reductions.append(result.cost_reduction_pct)

        cv = float(np.std(taxonomy.cost_levels) / np.mean(taxonomy.cost_levels))
        mean_reduction = float(np.mean(reductions)) if reductions else 0.0

        domain_results[domain] = {
            "cost_cv": cv,
            "mean_reduction_pct": mean_reduction,
        }

    # Check correlation between CV and reduction
    if len(domain_results) >= 3:
        cvs = [v["cost_cv"] for v in domain_results.values()]
        reds = [v["mean_reduction_pct"] for v in domain_results.values()]
        rho, p = spearmanr(cvs, reds)
    else:
        rho, p = 0.0, 1.0

    return {
        "per_domain": domain_results,
        "spearman_rho": rho,
        "p_value": p,
        "hypothesis_supported": rho > 0,
    }


def test_h5(metrics_df: pd.DataFrame) -> dict:
    """H5 — Robustesse de la métrique.

    Les classements par E[S] sont stables sous perturbation ±20% des coûts c_k.
    """
    results = {}

    for domain in metrics_df["domain"].unique():
        taxonomy = get_taxonomy(domain)
        df = metrics_df[metrics_df["domain"] == domain]

        if len(df) < 2:
            continue

        error_rates = {row["model"]: row["error_rate"] for _, row in df.iterrows()}
        severity_profiles = {}
        for _, row in df.iterrows():
            pi = np.array(row["severity_profile"])
            if pi.sum() == 0:
                pi = np.array([1.0, 0.0, 0.0, 0.0])
            severity_profiles[row["model"]] = pi

        sa = sensitivity_analysis(
            n_queries=10_000,
            error_rates=error_rates,
            cost_levels=taxonomy.cost_levels,
            severity_profiles=severity_profiles,
            perturbation=0.20,
            n_sim=50_000,
            seed=42,
        )

        results[domain] = {
            "min_spearman": sa["min_spearman"],
            "mean_spearman": float(np.mean(sa["spearman_correlations"])),
            "stable": sa["min_spearman"] >= 0.8,
        }

    n_stable = sum(1 for v in results.values() if v["stable"])
    return {
        "per_domain": results,
        "n_stable": n_stable,
        "hypothesis_supported": n_stable == len(results),
    }


def main():
    parser = argparse.ArgumentParser(description="Test all hypotheses")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    metrics_path = args.results_dir / "metrics.csv"
    if not metrics_path.exists():
        print(f"Metrics file not found at {metrics_path}")
        print("Run `python experiments/analysis.py` first.")
        return

    metrics_df = pd.read_csv(metrics_path)
    metrics_df["severity_profile"] = metrics_df["severity_profile"].apply(eval)

    print("=" * 70)
    print("HYPOTHESIS TESTING")
    print("=" * 70)

    # H1
    print("\nH1: Ranking Divergence (τ Kendall < 0.5)")
    h1 = test_h1(metrics_df)
    for domain, r in h1["per_domain"].items():
        status = "DIVERGENT" if r["divergent"] else "aligned"
        print(f"  {domain}: τ = {r['tau']:.3f}, p = {r['p_value']:.4f} [{status}]")
    print(
        f"  → H1 {'SUPPORTED' if h1['hypothesis_supported'] else 'NOT supported'} "
        f"({h1['n_divergent']} divergent domains)"
    )

    # H2
    print("\nH2: Severity Dominates Frequency")
    h2 = test_h2(metrics_df)
    for domain, r in h2["per_domain"].items():
        print(
            f"  {domain}: Var(π)={r['var_severity_only']:.0f}, "
            f"Var(p)={r['var_frequency_only']:.0f} "
            f"[{'severity' if r['severity_dominates'] else 'frequency'}]"
        )
    print(f"  → H2 {'SUPPORTED' if h2['hypothesis_supported'] else 'NOT supported'}")

    # H3
    print("\nH3: Ranking Inversions")
    h3 = test_h3(metrics_df)
    print(f"  Found {h3['n_inversions']} inversions")
    for inv in h3["inversions"][:5]:
        print(
            f"    {inv['domain']}: {inv['better_acc']} beats {inv['worse_acc']} "
            f"in accuracy (+{inv['acc_gap']:.3f}) but costs ${inv['loss_gap_dollars']:,.0f} more"
        )
    print(f"  → H3 {'SUPPORTED' if h3['hypothesis_supported'] else 'NOT supported'}")

    # H4
    print("\nH4: Routing Benefit ∝ Tail Heaviness")
    h4 = test_h4(metrics_df)
    for domain, r in h4["per_domain"].items():
        print(f"  {domain}: reduction={r['mean_reduction_pct']:.1f}%, CV={r['cost_cv']:.2f}")
    print(f"  Spearman ρ = {h4['spearman_rho']:.3f}, p = {h4['p_value']:.4f}")
    print(f"  → H4 {'SUPPORTED' if h4['hypothesis_supported'] else 'NOT supported'}")

    # H5
    print("\nH5: Robustness Under Cost Perturbation (±20%)")
    h5 = test_h5(metrics_df)
    for domain, r in h5["per_domain"].items():
        status = "STABLE" if r["stable"] else "unstable"
        print(f"  {domain}: min ρ={r['min_spearman']:.3f}, mean ρ={r['mean_spearman']:.3f} [{status}]")
    print(
        f"  → H5 {'SUPPORTED' if h5['hypothesis_supported'] else 'NOT supported'} "
        f"({h5['n_stable']}/{len(h5['per_domain'])} stable)"
    )

    # Save all results
    output_path = args.results_dir / "hypothesis_results.json"
    all_results = {"h1": h1, "h2": h2, "h3": h3, "h4": h4, "h5": h5}
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {output_path}")


if __name__ == "__main__":
    main()
