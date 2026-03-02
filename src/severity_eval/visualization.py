"""Visualization: figures for the paper."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from pathlib import Path
import numpy as np


def plot_aggregate_loss_distribution(
    S: np.ndarray,
    var: float,
    tvar: float,
    expected_loss: float,
    title: str = "Aggregate Loss Distribution",
    output_path: str | Path | None = None,
    n_bins: int = 80,
) -> plt.Figure:
    """Plot histogram of aggregate loss with VaR/TVaR markers.

    Parameters
    ----------
    S : ndarray
        Simulated aggregate losses.
    var : float
        Value-at-Risk.
    tvar : float
        Tail Value-at-Risk.
    expected_loss : float
        Expected loss E[S].
    title : str
        Figure title.
    output_path : str or Path or None
        If provided, save figure to this path.
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(S, bins=n_bins, density=True, alpha=0.7, color="#4C72B0", edgecolor="white")
    ax.axvline(expected_loss, color="#C44E52", linestyle="--", linewidth=2, label=f"E[S] = {expected_loss:,.0f}")
    ax.axvline(var, color="#DD8452", linestyle="-.", linewidth=2, label=f"VaR = {var:,.0f}")
    ax.axvline(tvar, color="#55A868", linestyle=":", linewidth=2, label=f"TVaR = {tvar:,.0f}")

    ax.set_xlabel("Aggregate Loss ($)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.ticklabel_format(style="plain", axis="x")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_ranking_divergence(
    models: list[str],
    accuracy_ranks: list[int],
    expected_loss_ranks: list[int],
    title: str = "Ranking Divergence: Accuracy vs E[S]",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Bump chart showing ranking divergence between accuracy and E[S].

    Parameters
    ----------
    models : list of str
        Model names.
    accuracy_ranks : list of int
        Rank by accuracy (1 = best).
    expected_loss_ranks : list of int
        Rank by E[S] (1 = lowest loss = best).
    title : str
        Figure title.
    output_path : str or Path or None
        If provided, save figure.

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for i, model in enumerate(models):
        ax.plot(
            [0, 1],
            [accuracy_ranks[i], expected_loss_ranks[i]],
            "o-",
            color=colors[i],
            linewidth=2,
            markersize=8,
            label=model,
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Accuracy Rank", "E[S] Rank"])
    ax.set_ylabel("Rank (1 = best)")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_severity_profiles(
    models: list[str],
    profiles: dict[str, np.ndarray],
    labels: list[str],
    title: str = "Severity Profiles by Model",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Stacked bar chart of severity profiles.

    Parameters
    ----------
    models : list of str
        Model names.
    profiles : dict
        {model_name: severity_profile array}.
    labels : list of str
        Severity level labels.
    title : str
        Figure title.
    output_path : str or Path or None
        If provided, save figure.

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(models))
    width = 0.6
    colors = ["#4C72B0", "#DD8452", "#C44E52", "#8172B3"]

    bottom = np.zeros(len(models))
    for j, label in enumerate(labels):
        values = [profiles[m][j] for m in models]
        ax.bar(x, values, width, bottom=bottom, label=label, color=colors[j % len(colors)])
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Proportion")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_routing_impact(
    domains: list[str],
    expected_loss: list[float],
    expected_cost_routed: list[float],
    title: str = "Impact of HITL Routing by Domain",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart comparing E[S] vs E[C] per domain.

    Parameters
    ----------
    domains : list of str
        Domain names.
    expected_loss : list of float
        E[S] per domain.
    expected_cost_routed : list of float
        E[C] after routing per domain.
    title : str
        Figure title.
    output_path : str or Path or None
        If provided, save figure.

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(domains))
    width = 0.35

    ax.bar(x - width / 2, expected_loss, width, label="E[S] (no routing)", color="#C44E52", alpha=0.8)
    ax.bar(x + width / 2, expected_cost_routed, width, label="E[C] (with routing)", color="#4C72B0", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=30, ha="right")
    ax.set_ylabel("Expected Cost ($)")
    ax.set_title(title)
    ax.legend()
    ax.ticklabel_format(style="plain", axis="y")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig
