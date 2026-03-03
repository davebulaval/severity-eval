"""Statistical analysis: compute metrics, test hypotheses, generate tables.

Usage:
    python experiments/analysis.py --results-dir experiments/results --output results/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from severity_eval.compound_loss import simulate_aggregate_loss
from severity_eval.risk_measures import bootstrap_ci, compute_risk_measures
from severity_eval.routing import analyze_routing
from severity_eval.taxonomy import get_taxonomy, severity_label_to_index


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load all evaluation results into a single DataFrame."""
    dfs = []
    for f in sorted(results_dir.glob("*.json")):
        df = pd.read_json(f, orient="records")
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No result files found in {results_dir}")
    return pd.concat(dfs, ignore_index=True)


def compute_model_metrics(
    df: pd.DataFrame,
    domain: str,
    n_queries: int = 10_000,
    n_sim: int = 100_000,
    alpha: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute actuarial metrics per model for a given domain.

    Parameters
    ----------
    df : DataFrame
        Results with columns: model, answer, prediction, severity, domain.
    domain : str
        Domain to filter on.
    n_queries : int
        Queries per period.
    n_sim : int
        Monte Carlo replications.
    alpha : float
        VaR/TVaR confidence level.
    seed : int
        Random seed.

    Returns
    -------
    DataFrame with one row per model.
    """
    taxonomy = get_taxonomy(domain)
    domain_df = df[df["domain"] == domain].copy()

    records = []
    for model, group in domain_df.groupby("model"):
        n = len(group)
        # Use pre-computed 'correct' column from the scoring pipeline
        correct = group["correct"].sum()
        accuracy = correct / n
        error_rate = 1 - accuracy

        # Severity profile from errors
        errors = group[~group["correct"]]
        if len(errors) == 0:
            records.append(
                {
                    "model": model,
                    "domain": domain,
                    "n": n,
                    "accuracy": accuracy,
                    "error_rate": 0.0,
                    "expected_loss": 0.0,
                    "var": 0.0,
                    "tvar": 0.0,
                    "severity_profile": [0.0] * len(taxonomy.cost_levels),
                }
            )
            continue

        counts = np.zeros(len(taxonomy.cost_levels))
        for sev in errors["severity"]:
            idx = severity_label_to_index(sev)
            counts[idx] += 1
        pi = counts / counts.sum()

        # Simulate
        S = simulate_aggregate_loss(
            n_queries,
            error_rate,
            taxonomy.cost_levels,
            pi,
            n_sim,
            seed=seed,
        )
        measures = compute_risk_measures(S, alpha=alpha)
        ci_lo, ci_hi = bootstrap_ci(S, statistic="expected_loss", seed=seed)

        records.append(
            {
                "model": model,
                "domain": domain,
                "n": n,
                "accuracy": accuracy,
                "error_rate": error_rate,
                "severity_profile": pi.tolist(),
                "expected_loss": measures["expected_loss"],
                "expected_loss_ci_lo": ci_lo,
                "expected_loss_ci_hi": ci_hi,
                "var": measures["var"],
                "tvar": measures["tvar"],
            }
        )

    return pd.DataFrame(records)


def test_h1_ranking_divergence(metrics_df: pd.DataFrame) -> dict:
    """H1: τ Kendall between accuracy rank and E[S] rank."""
    results = {}
    for domain in metrics_df["domain"].unique():
        domain_metrics = metrics_df[metrics_df["domain"] == domain].copy()
        if len(domain_metrics) < 3:
            continue

        # Rank by accuracy (higher = better → rank 1)
        domain_metrics["accuracy_rank"] = domain_metrics["accuracy"].rank(ascending=False)
        # Rank by E[S] (lower = better → rank 1)
        domain_metrics["loss_rank"] = domain_metrics["expected_loss"].rank(ascending=True)

        tau, p_value = kendalltau(
            domain_metrics["accuracy_rank"],
            domain_metrics["loss_rank"],
        )
        results[domain] = {
            "tau": tau,
            "p_value": p_value,
            "divergent": tau < 0.5 and p_value < 0.05,
            "n_models": len(domain_metrics),
        }
    return results


def test_h3_inversions(metrics_df: pd.DataFrame) -> list[dict]:
    """H3: Find pairs where accuracy(A) > accuracy(B) but E[S](A) > E[S](B)."""
    inversions = []
    for domain in metrics_df["domain"].unique():
        domain_metrics = metrics_df[metrics_df["domain"] == domain]
        models = domain_metrics.to_dict("records")

        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a, b = models[i], models[j]
                if a["accuracy"] > b["accuracy"] and a["expected_loss"] > b["expected_loss"]:
                    inversions.append(
                        {
                            "domain": domain,
                            "model_a": a["model"],
                            "model_b": b["model"],
                            "accuracy_a": a["accuracy"],
                            "accuracy_b": b["accuracy"],
                            "loss_a": a["expected_loss"],
                            "loss_b": b["expected_loss"],
                            "loss_gap": a["expected_loss"] - b["expected_loss"],
                        }
                    )
                elif b["accuracy"] > a["accuracy"] and b["expected_loss"] > a["expected_loss"]:
                    inversions.append(
                        {
                            "domain": domain,
                            "model_a": b["model"],
                            "model_b": a["model"],
                            "accuracy_a": b["accuracy"],
                            "accuracy_b": a["accuracy"],
                            "loss_a": b["expected_loss"],
                            "loss_b": a["expected_loss"],
                            "loss_gap": b["expected_loss"] - a["expected_loss"],
                        }
                    )
    return inversions


def test_h4_routing_benefit(metrics_df: pd.DataFrame, n_queries: int = 10_000) -> dict:
    """H4: Routing benefit proportional to tail heaviness."""
    results = {}
    for domain in metrics_df["domain"].unique():
        taxonomy = get_taxonomy(domain)
        domain_metrics = metrics_df[metrics_df["domain"] == domain]

        reductions = []
        for _, row in domain_metrics.iterrows():
            pi = np.array(row["severity_profile"])
            if row["error_rate"] == 0:
                continue
            routing = analyze_routing(
                n_queries=n_queries,
                error_rate=row["error_rate"],
                cost_levels=taxonomy.cost_levels,
                severity_profile=pi,
                retention_threshold=taxonomy.cost_levels[2],  # route major + critical
            )
            reductions.append(routing.cost_reduction_pct)

        cv = np.std(taxonomy.cost_levels) / np.mean(taxonomy.cost_levels)
        results[domain] = {
            "mean_reduction_pct": np.mean(reductions) if reductions else 0.0,
            "cost_cv": cv,
        }
    return results


def generate_latex_table(metrics_df: pd.DataFrame, output_path: Path):
    """Generate LaTeX table of main results."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Model performance by domain: accuracy vs.\ expected loss ranking.}",
        r"\label{tab:main-results}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Domain & Model & Acc.\% & $\hat{p}$ & E[S] (\$) & VaR$_{95}$ (\$) & TVaR$_{95}$ (\$) \\",
        r"\midrule",
    ]

    for domain in sorted(metrics_df["domain"].unique()):
        domain_df = metrics_df[metrics_df["domain"] == domain].sort_values("expected_loss")
        first = True
        for _, row in domain_df.iterrows():
            d = domain.replace("_", r"\_") if first else ""
            lines.append(
                f"  {d} & {row['model']} & {row['accuracy'] * 100:.1f} & "
                f"{row['error_rate']:.3f} & {row['expected_loss']:,.0f} & "
                f"{row['var']:,.0f} & {row['tvar']:,.0f} \\\\"
            )
            first = False
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.extend([r"\end{tabular}", r"\end{table}"])

    output_path.write_text("\n".join(lines))
    print(f"LaTeX table saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation results")
    parser.add_argument("--results-dir", type=Path, default=Path("experiments/results"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    df = load_results(args.results_dir)
    print(f"Loaded {len(df)} total predictions")

    # Compute metrics per (model × domain)
    all_metrics = []
    for domain in df["domain"].unique():
        print(f"\nComputing metrics for {domain}...")
        metrics = compute_model_metrics(df, domain)
        all_metrics.append(metrics)
        print(metrics[["model", "accuracy", "expected_loss", "var", "tvar"]].to_string())

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    metrics_df.to_csv(args.output / "metrics.csv", index=False)

    # H1: Ranking divergence
    print("\n" + "=" * 60)
    print("H1: Ranking Divergence (τ Kendall)")
    print("=" * 60)
    h1 = test_h1_ranking_divergence(metrics_df)
    for domain, result in h1.items():
        status = "DIVERGENT" if result["divergent"] else "aligned"
        print(f"  {domain}: τ = {result['tau']:.3f}, p = {result['p_value']:.4f} [{status}]")

    # H3: Inversions
    print("\n" + "=" * 60)
    print("H3: Ranking Inversions")
    print("=" * 60)
    inversions = test_h3_inversions(metrics_df)
    print(f"  Found {len(inversions)} inversions")
    for inv in inversions:
        print(
            f"  {inv['domain']}: {inv['model_a']} (acc={inv['accuracy_a']:.3f}, "
            f"E[S]=${inv['loss_a']:,.0f}) vs {inv['model_b']} (acc={inv['accuracy_b']:.3f}, "
            f"E[S]=${inv['loss_b']:,.0f}) — gap=${inv['loss_gap']:,.0f}"
        )

    # H4: Routing benefit
    print("\n" + "=" * 60)
    print("H4: Routing Benefit by Domain")
    print("=" * 60)
    h4 = test_h4_routing_benefit(metrics_df)
    for domain, result in h4.items():
        print(f"  {domain}: mean reduction={result['mean_reduction_pct']:.1f}%, cost CV={result['cost_cv']:.2f}")

    # Save all results
    results = {
        "h1_ranking_divergence": h1,
        "h3_inversions": inversions,
        "h4_routing_benefit": h4,
    }
    with open(args.output / "hypothesis_tests.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Generate LaTeX table
    generate_latex_table(metrics_df, args.output / "table_main.tex")

    print(f"\nAll results saved to {args.output}/")


if __name__ == "__main__":
    main()
