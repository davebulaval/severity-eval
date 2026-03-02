"""Generate all paper figures from analysis results.

Usage:
    python experiments/figures.py --results-dir results --output paper/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from severity_eval.compound_loss import simulate_aggregate_loss
from severity_eval.risk_measures import compute_risk_measures
from severity_eval.routing import analyze_routing
from severity_eval.taxonomy import SEVERITY_LABELS, get_taxonomy
from severity_eval.visualization import (
    plot_aggregate_loss_distribution,
    plot_ranking_divergence,
    plot_routing_impact,
    plot_severity_profiles,
)

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 300,
    }
)


def figure_1_aggregate_loss(metrics_df: pd.DataFrame, output_dir: Path):
    """Figure 1: Aggregate loss distribution for a representative model/domain."""
    # Use finance domain, pick the model with highest E[S]
    finance = metrics_df[metrics_df["domain"] == "finance"]
    if finance.empty:
        print("  No finance data, skipping Figure 1")
        return

    worst = finance.loc[finance["expected_loss"].idxmax()]
    taxonomy = get_taxonomy("finance")
    pi = np.array(worst["severity_profile"])

    S = simulate_aggregate_loss(
        n_queries=10_000,
        error_rate=worst["error_rate"],
        cost_levels=taxonomy.cost_levels,
        severity_profile=pi,
        n_sim=200_000,
        seed=42,
    )
    measures = compute_risk_measures(S, alpha=0.95)

    fig = plot_aggregate_loss_distribution(
        S,
        var=measures["var"],
        tvar=measures["tvar"],
        expected_loss=measures["expected_loss"],
        title=f"Aggregate Loss Distribution — {worst['model']} (Finance)",
        output_path=output_dir / "fig1_aggregate_loss.pdf",
    )
    plt.close(fig)
    print("  Figure 1: aggregate loss distribution")


def figure_2_ranking_divergence(metrics_df: pd.DataFrame, output_dir: Path):
    """Figure 2: Bump chart of ranking divergence per domain."""
    for domain in metrics_df["domain"].unique():
        domain_df = metrics_df[metrics_df["domain"] == domain].copy()
        if len(domain_df) < 2:
            continue

        models = domain_df["model"].tolist()
        acc_ranks = domain_df["accuracy"].rank(ascending=False).astype(int).tolist()
        loss_ranks = domain_df["expected_loss"].rank(ascending=True).astype(int).tolist()

        fig = plot_ranking_divergence(
            models=models,
            accuracy_ranks=acc_ranks,
            expected_loss_ranks=loss_ranks,
            title=f"Ranking Divergence — {domain.replace('_', ' ').title()}",
            output_path=output_dir / f"fig2_ranking_{domain}.pdf",
        )
        plt.close(fig)
    print("  Figure 2: ranking divergence bump charts")


def figure_3_severity_profiles(metrics_df: pd.DataFrame, output_dir: Path):
    """Figure 3: Stacked bar chart of severity profiles per domain."""
    for domain in metrics_df["domain"].unique():
        domain_df = metrics_df[metrics_df["domain"] == domain]
        models = domain_df["model"].tolist()
        profiles = {row["model"]: np.array(row["severity_profile"]) for _, row in domain_df.iterrows()}

        fig = plot_severity_profiles(
            models=models,
            profiles=profiles,
            labels=SEVERITY_LABELS,
            title=f"Severity Profiles — {domain.replace('_', ' ').title()}",
            output_path=output_dir / f"fig3_profiles_{domain}.pdf",
        )
        plt.close(fig)
    print("  Figure 3: severity profiles")


def figure_4_routing_impact(metrics_df: pd.DataFrame, output_dir: Path):
    """Figure 4: Bar chart comparing E[S] vs E[C] per domain."""
    domains = []
    es_values = []
    ec_values = []

    for domain in sorted(metrics_df["domain"].unique()):
        taxonomy = get_taxonomy(domain)
        domain_df = metrics_df[metrics_df["domain"] == domain]

        # Average across models
        mean_error = domain_df["error_rate"].mean()
        mean_pi = np.mean([np.array(row["severity_profile"]) for _, row in domain_df.iterrows()], axis=0)
        # Renormalize in case
        if mean_pi.sum() > 0:
            mean_pi = mean_pi / mean_pi.sum()
        else:
            continue

        routing = analyze_routing(
            n_queries=10_000,
            error_rate=mean_error,
            cost_levels=taxonomy.cost_levels,
            severity_profile=mean_pi,
            retention_threshold=taxonomy.cost_levels[2],  # route major + critical
        )

        domains.append(domain.replace("_", " ").title())
        es_values.append(routing.expected_loss_unrouted)
        ec_values.append(routing.expected_total_cost)

    if domains:
        fig = plot_routing_impact(
            domains=domains,
            expected_loss=es_values,
            expected_cost_routed=ec_values,
            title="Impact of HITL Routing by Domain",
            output_path=output_dir / "fig4_routing_impact.pdf",
        )
        plt.close(fig)
        print("  Figure 4: routing impact")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    metrics_path = args.results_dir / "metrics.csv"
    if not metrics_path.exists():
        print(f"Metrics file not found at {metrics_path}")
        print("Run `python experiments/analysis.py` first.")
        return

    metrics_df = pd.read_csv(metrics_path)
    # Parse severity_profile from string representation
    metrics_df["severity_profile"] = metrics_df["severity_profile"].apply(eval)

    print("Generating figures...")
    figure_1_aggregate_loss(metrics_df, args.output)
    figure_2_ranking_divergence(metrics_df, args.output)
    figure_3_severity_profiles(metrics_df, args.output)
    figure_4_routing_impact(metrics_df, args.output)

    print(f"\nAll figures saved to {args.output}/")


if __name__ == "__main__":
    main()
