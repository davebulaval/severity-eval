"""Tests for severity_eval.visualization.

These are smoke tests for matplotlib plotting functions: they verify that
each function returns a real Figure, that the figure contains the expected
content (axis labels, legend entries, vertical markers at the right
positions), and that the output_path file is actually written.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend for CI

import numpy as np
import pytest
from matplotlib.figure import Figure

from severity_eval.visualization import (
    plot_aggregate_loss_distribution,
    plot_ranking_divergence,
    plot_routing_impact,
    plot_severity_profiles,
)


# ----------------------------------------------------------------------------
# plot_aggregate_loss_distribution
# ----------------------------------------------------------------------------


def test_aggregate_loss_distribution_returns_figure_with_var_tvar_markers() -> None:
    rng = np.random.default_rng(0)
    s = rng.gamma(shape=2.0, scale=1000.0, size=10_000)
    var, tvar = float(np.quantile(s, 0.95)), float(s[s > np.quantile(s, 0.95)].mean())
    expected_loss = float(s.mean())

    fig = plot_aggregate_loss_distribution(s, var, tvar, expected_loss)

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    # vertical lines (axvline) are stored as Line2D; their x coords match var,
    # tvar, expected_loss
    vline_xs = sorted(
        {line.get_xdata()[0] for line in ax.lines if len(line.get_xdata()) == 2}
    )
    assert any(abs(x - expected_loss) < 1e-6 for x in vline_xs), vline_xs
    assert any(abs(x - var) < 1e-6 for x in vline_xs), vline_xs
    assert any(abs(x - tvar) < 1e-6 for x in vline_xs), vline_xs


def test_aggregate_loss_distribution_saves_pdf(tmp_path) -> None:
    s = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    output = tmp_path / "agg_loss.pdf"

    plot_aggregate_loss_distribution(s, 450.0, 475.0, 300.0, output_path=output)

    assert output.exists()
    assert output.stat().st_size > 1000  # non-trivial PDF


# ----------------------------------------------------------------------------
# plot_ranking_divergence
# ----------------------------------------------------------------------------


def test_ranking_divergence_one_line_per_model() -> None:
    models = ["gpt-oss-20b", "qwen3-14b", "deepseek-r1-70b"]
    acc_ranks = [1, 2, 3]
    es_ranks = [3, 2, 1]  # full reversal

    fig = plot_ranking_divergence(models, acc_ranks, es_ranks)

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    # one Line2D per model
    assert len(ax.lines) == len(models), ax.lines
    # legend has one entry per model
    legend = ax.get_legend()
    assert legend is not None
    assert {t.get_text() for t in legend.get_texts()} == set(models)
    # y-axis inverted (rank 1 at top)
    ymin, ymax = ax.get_ylim()
    assert ymin > ymax, (ymin, ymax)


def test_ranking_divergence_endpoints_match_inputs() -> None:
    models = ["a", "b"]
    acc_ranks = [1, 2]
    es_ranks = [2, 1]

    fig = plot_ranking_divergence(models, acc_ranks, es_ranks)
    ax = fig.axes[0]

    # First line goes from (0, acc=1) to (1, es=2); second from (0, 2) to (1, 1).
    line_a = ax.lines[0]
    assert list(line_a.get_xdata()) == [0, 1]
    assert list(line_a.get_ydata()) == [1, 2]
    line_b = ax.lines[1]
    assert list(line_b.get_ydata()) == [2, 1]


# ----------------------------------------------------------------------------
# plot_severity_profiles
# ----------------------------------------------------------------------------


def test_severity_profiles_stacked_bar_proportions_sum_to_one() -> None:
    models = ["m1", "m2"]
    profiles = {
        "m1": np.array([0.7, 0.2, 0.05, 0.05]),
        "m2": np.array([0.1, 0.3, 0.4, 0.2]),
    }
    labels = ["negl.", "minor", "major", "critical"]

    fig = plot_severity_profiles(models, profiles, labels)

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    # 4 BarContainers, one per severity level
    bar_containers = ax.containers
    assert len(bar_containers) == 4

    # Sum of bar heights at each x must equal 1.0 (proportions)
    for i, model in enumerate(models):
        total = sum(c[i].get_height() for c in bar_containers)
        assert abs(total - 1.0) < 1e-9, (model, total)


def test_severity_profiles_with_three_categories_no_color_overrun() -> None:
    # K=3 must not crash even though default colors list has 4 entries
    models = ["m"]
    profiles = {"m": np.array([0.5, 0.3, 0.2])}
    labels = ["low", "med", "high"]

    fig = plot_severity_profiles(models, profiles, labels)

    assert isinstance(fig, Figure)
    assert len(fig.axes[0].containers) == 3


# ----------------------------------------------------------------------------
# plot_routing_impact
# ----------------------------------------------------------------------------


def test_routing_impact_two_bars_per_domain() -> None:
    domains = ["finance", "medical", "legal"]
    es = [30e6, 600e6, 100e6]
    ec = [5e6, 50e6, 2e6]

    fig = plot_routing_impact(domains, es, ec)

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    containers = ax.containers
    assert len(containers) == 2  # E[S] bars + E[C] bars
    es_bars = containers[0]
    ec_bars = containers[1]
    assert len(es_bars) == len(domains)
    assert len(ec_bars) == len(domains)
    # heights match the inputs
    for i, expected in enumerate(es):
        assert es_bars[i].get_height() == pytest.approx(expected)
    for i, expected in enumerate(ec):
        assert ec_bars[i].get_height() == pytest.approx(expected)
    # legend has both labels
    legend_texts = {t.get_text() for t in ax.get_legend().get_texts()}
    assert "E[S] (no routing)" in legend_texts
    assert "E[C] (with routing)" in legend_texts


def test_routing_impact_saves_to_path(tmp_path) -> None:
    output = tmp_path / "routing.pdf"
    plot_routing_impact(["d"], [1.0], [0.5], output_path=output)

    assert output.exists()
    assert output.stat().st_size > 1000
