"""Public API for severity-eval.

This is the main entry point for evaluating models. It produces a
SeverityReport with summary tables, risk measures, and figures.

Usage
-----
>>> import severity_eval
>>> report = severity_eval.evaluate(
...     predictions=["Paris", "London", "wrong", "Berlin"],
...     references=["Paris", "London", "Madrid", "Berlin"],
...     severity_annotations=["negligible", "minor", "critical", "negligible"],
...     cost_levels=[100, 1000, 10000, 100000],
... )
>>> print(report)
>>> report.plot()
>>> report.to_dataframe()
>>> report.to_latex()
"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np

from severity_eval.compound_loss import simulate_aggregate_loss
from severity_eval.risk_measures import bootstrap_ci, compute_risk_measures
from severity_eval.routing import RoutingResult, analyze_routing

# Default labels when K=4 (convenience for common case)
_DEFAULT_LABELS_4 = ["negligible", "minor", "major", "critical"]


@dataclass
class SeverityReport:
    """Full evaluation report with risk measures, tables, and figures.

    Attributes
    ----------
    cost_levels : np.ndarray
        Dollar cost for each severity level (c_1, …, c_K).
    labels : list of str
        Severity level names.
    accuracy : float
        Fraction of correct predictions.
    error_rate : float
        1 - accuracy.
    n_samples : int
        Total instances evaluated.
    n_errors : int
        Incorrect predictions.
    severity_profile : np.ndarray
        π_k = P(severity=k | error).
    severity_counts : np.ndarray
        Raw error count per level.
    losses : np.ndarray
        Simulated aggregate losses (n_sim values).
    expected_loss : float
        E[S].
    expected_loss_ci : tuple[float, float]
        95% bootstrap CI for E[S].
    var : float
        VaR at level alpha.
    tvar : float
        TVaR at level alpha.
    alpha : float
        Confidence level for VaR/TVaR.
    n_queries : int
        Queries per period in simulation.
    n_sim : int
        Monte Carlo replications.
    routing : RoutingResult | None
        HITL routing analysis (if requested).
    """

    cost_levels: np.ndarray
    labels: list[str]
    accuracy: float
    error_rate: float
    n_samples: int
    n_errors: int
    severity_profile: np.ndarray
    severity_counts: np.ndarray
    losses: np.ndarray
    expected_loss: float
    expected_loss_ci: tuple[float, float]
    var: float
    tvar: float
    alpha: float
    n_queries: int
    n_sim: int
    routing: RoutingResult | None = None

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return self.summary()

    def summary(self) -> str:
        """Human-readable summary table."""
        buf = StringIO()
        w = buf.write

        w(f"{'=' * 62}\n")
        w("  Severity Report\n")
        w(f"{'=' * 62}\n\n")

        # Section 1: frequency
        w("  Frequency\n")
        w(f"  {'-' * 40}\n")
        w(f"  Samples evaluated     {self.n_samples:>12,}\n")
        w(f"  Correct               {self.n_samples - self.n_errors:>12,}\n")
        w(f"  Errors                {self.n_errors:>12,}\n")
        w(f"  Accuracy              {self.accuracy:>12.1%}\n")
        w(f"  Error rate (p)        {self.error_rate:>12.4f}\n")
        w("\n")

        # Section 2: severity profile with c_k and π_k
        w("  Severity profile (given error)\n")
        w(f"  {'Label':<14s}  {'c_k':>10s}   {'pi_k':>6s}  {'errors':>8s}\n")
        w(f"  {'-' * 50}\n")
        for i, (label, cost) in enumerate(zip(self.labels, self.cost_levels, strict=True)):
            cnt = int(self.severity_counts[i])
            pi = self.severity_profile[i]
            w(f"  {label:<14s}  ${cost:>10,.0f}   {pi:>6.1%}  {cnt:>7,}\n")
        mu_X = float((self.cost_levels * self.severity_profile).sum())
        w(f"  {'-' * 50}\n")
        w(f"  {'mu_X':<14s}  ${mu_X:>10,.0f}\n")
        w("\n")

        # Section 3: risk measures
        ci_lo, ci_hi = self.expected_loss_ci
        w(f"  Risk measures (n={self.n_queries:,}, {self.n_sim:,} MC sims)\n")
        w(f"  {'-' * 40}\n")
        w(f"  E[S]                  ${self.expected_loss:>12,.0f}\n")
        w(f"  95% CI                ${ci_lo:>12,.0f} — ${ci_hi:,.0f}\n")
        w(f"  VaR_{self.alpha:.0%}             ${self.var:>12,.0f}\n")
        w(f"  TVaR_{self.alpha:.0%}            ${self.tvar:>12,.0f}\n")
        w("\n")

        # Section 4: routing (if computed)
        if self.routing is not None:
            r = self.routing
            w(f"  HITL routing (threshold=${r.retention_threshold:,})\n")
            w(f"  {'-' * 55}\n")

            w(f"  Retained (AI) — {len(r.retained_cost_levels)} levels, mu_X_ret=${r.mu_X_retained:,.0f}\n")
            for c_k, pi_k in zip(r.retained_cost_levels, r.retained_severity_profile, strict=True):
                w(f"    c_k=${c_k:>10,.0f}   pi_k={pi_k:>6.1%}\n")

            w(f"  Routed (human) — {len(r.routed_cost_levels)} levels, h=${r.human_review_cost:,}/query\n")
            for c_k, pi_k in zip(r.routed_cost_levels, r.routed_severity_profile, strict=True):
                w(f"    c_k=${c_k:>10,.0f}   pi_k={pi_k:>6.1%}\n")

            w("\n")
            w(f"  Routing ratio (rho)   {r.routing_ratio:>12.1%}\n")
            w(f"  Queries routed        {r.n_routed:>12,}\n")
            w(f"  mu_X (overall)        ${r.mu_X:>12,.0f}\n")
            w(f"  mu_X (retained)       ${r.mu_X_retained:>12,.0f}\n")
            w(f"  E[S] without routing  ${r.expected_loss_unrouted:>12,.0f}\n")
            w(f"  E[S_ret]              ${r.expected_loss_retained:>12,.0f}\n")
            w(f"  Routing cost          ${r.routing_cost:>12,.0f}\n")
            w(f"  E[C] with routing     ${r.expected_total_cost:>12,.0f}\n")
            w(f"  Cost reduction        {r.cost_reduction_pct:>12.1f}%\n")
            w("\n")

        w(f"{'=' * 62}\n")
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Structured output
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Export as flat dictionary (JSON-serializable)."""
        d = {
            "cost_levels": self.cost_levels.tolist(),
            "labels": self.labels,
            "accuracy": self.accuracy,
            "error_rate": self.error_rate,
            "n_samples": self.n_samples,
            "n_errors": self.n_errors,
            "severity_profile": self.severity_profile.tolist(),
            "severity_counts": self.severity_counts.tolist(),
            "expected_loss": self.expected_loss,
            "expected_loss_ci": list(self.expected_loss_ci),
            "var": self.var,
            "tvar": self.tvar,
            "alpha": self.alpha,
            "n_queries": self.n_queries,
            "n_sim": self.n_sim,
        }
        if self.routing is not None:
            r = self.routing
            d["routing"] = {
                "threshold": r.retention_threshold,
                "human_review_cost": r.human_review_cost,
                "routing_ratio": r.routing_ratio,
                "mu_X": r.mu_X,
                "mu_X_retained": r.mu_X_retained,
                "retained_cost_levels": r.retained_cost_levels.tolist(),
                "retained_severity_profile": r.retained_severity_profile.tolist(),
                "routed_cost_levels": r.routed_cost_levels.tolist(),
                "routed_severity_profile": r.routed_severity_profile.tolist(),
                "n_routed": r.n_routed,
                "n_retained": r.n_retained,
                "expected_loss_unrouted": r.expected_loss_unrouted,
                "expected_loss_retained": r.expected_loss_retained,
                "routing_cost": r.routing_cost,
                "expected_total_cost": r.expected_total_cost,
                "cost_reduction_pct": r.cost_reduction_pct,
            }
        return d

    def to_dataframe(self) -> dict:
        """Export risk measures as a pandas DataFrame."""
        import pandas as pd

        rows = []
        for i, (label, cost) in enumerate(zip(self.labels, self.cost_levels, strict=True)):
            rows.append(
                {
                    "severity": label,
                    "cost_level": cost,
                    "error_count": int(self.severity_counts[i]),
                    "probability": self.severity_profile[i],
                }
            )

        profile_df = pd.DataFrame(rows)

        summary_df = pd.DataFrame(
            [
                {
                    "accuracy": self.accuracy,
                    "error_rate": self.error_rate,
                    "E[S]": self.expected_loss,
                    "E[S] CI low": self.expected_loss_ci[0],
                    "E[S] CI high": self.expected_loss_ci[1],
                    f"VaR_{self.alpha:.0%}": self.var,
                    f"TVaR_{self.alpha:.0%}": self.tvar,
                }
            ]
        )

        return {"summary": summary_df, "severity_profile": profile_df}

    def to_latex(self) -> str:
        """Export as LaTeX table fragment."""
        lines = []
        lines.append(r"\begin{tabular}{lrrrr}")
        lines.append(r"\toprule")
        lines.append(r"Severity & $c_k$ (\$) & Count & $\hat{\pi}_k$ \\")
        lines.append(r"\midrule")

        for i, (label, cost) in enumerate(zip(self.labels, self.cost_levels, strict=True)):
            cnt = int(self.severity_counts[i])
            pi = self.severity_profile[i]
            lines.append(f"  {label.capitalize()} & {cost:,.0f} & {cnt:,} & {pi:.3f} \\\\")

        lines.append(r"\midrule")
        lines.append(
            f"  \\multicolumn{{4}}{{l}}"
            f"{{$E[S] = \\${self.expected_loss:,.0f}$, "
            f"$\\mathrm{{VaR}}_{{{self.alpha * 100:.0f}\\%}} = \\${self.var:,.0f}$, "
            f"$\\mathrm{{TVaR}}_{{{self.alpha * 100:.0f}\\%}} = \\${self.tvar:,.0f}$}} \\\\"
        )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    def plot(self, output_dir: str | Path | None = None) -> dict:
        """Generate all figures. Returns dict of {name: matplotlib.Figure}.

        Parameters
        ----------
        output_dir : str or Path or None
            If provided, save all figures as PDF to this directory.
        """
        import matplotlib.pyplot as plt

        from severity_eval.visualization import plot_aggregate_loss_distribution

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        figures = {}

        # Figure 1: Aggregate loss distribution
        fig_path = (output_dir / "loss_distribution.pdf") if output_dir else None
        fig1 = plot_aggregate_loss_distribution(
            S=self.losses,
            var=self.var,
            tvar=self.tvar,
            expected_loss=self.expected_loss,
            title="Aggregate Loss Distribution",
            output_path=fig_path,
        )
        figures["loss_distribution"] = fig1

        # Figure 2: Severity profile bar chart
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        colors = ["#4C72B0", "#DD8452", "#C44E52", "#8172B3"]
        x = np.arange(len(self.labels))

        bars = ax2.bar(x, self.severity_profile, color=colors[: len(self.labels)], edgecolor="white", width=0.6)
        ax2.set_xticks(x)
        ax2.set_xticklabels(
            [f"{lbl}\n(${c:,.0f})" for lbl, c in zip(self.labels, self.cost_levels, strict=True)],
            fontsize=9,
        )
        ax2.set_ylabel(r"$\pi_k$ = P(severity = k | error)")
        ax2.set_title("Severity Profile")
        ax2.set_ylim(0, max(self.severity_profile) * 1.25 if max(self.severity_profile) > 0 else 1)

        for bar, pi in zip(bars, self.severity_profile, strict=True):
            if pi > 0:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{pi:.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        fig2.tight_layout()
        if output_dir:
            fig2.savefig(output_dir / "severity_profile.pdf", dpi=300, bbox_inches="tight")
        figures["severity_profile"] = fig2

        return figures


# ======================================================================
# Main evaluate() function
# ======================================================================


def evaluate(
    predictions: list,
    references: list,
    severity_annotations: list[str],
    cost_levels: list[int] | list[float] | np.ndarray,
    n_queries: int = 10000,
    n_sim: int = 100000,
    alpha: float = 0.95,
    seed: int = 42,
    labels: list[str] | None = None,
    routing_threshold: int | float | None = None,
    human_review_cost: int | float = 50,
) -> SeverityReport:
    """Evaluate a model and produce a full severity report.

    Parameters
    ----------
    predictions : list
        Model outputs.
    references : list
        Ground truth answers.
    severity_annotations : list of str
        Severity label per instance. Must match ``labels``.
    cost_levels : list of int/float or ndarray
        Dollar cost for each severity level (c_1, …, c_K).
        Example: [100, 1000, 10000, 100000]
    n_queries : int
        Number of queries per period for MC simulation.
    n_sim : int
        Number of Monte Carlo replications.
    alpha : float
        Confidence level for VaR/TVaR (e.g. 0.95).
    seed : int
        Random seed for reproducibility.
    labels : list of str or None
        Names for each severity level. Defaults to
        ['negligible', 'minor', 'major', 'critical'].
    routing_threshold : int, float, or None
        If set, compute HITL routing analysis. Errors with
        c_k >= threshold are routed to human review.
    human_review_cost : int or float
        Cost per human-reviewed query (h).

    Returns
    -------
    SeverityReport

    Examples
    --------
    >>> report = evaluate(
    ...     predictions=["Paris", "London", "wrong"],
    ...     references=["Paris", "London", "Madrid"],
    ...     severity_annotations=["negligible", "minor", "critical"],
    ...     cost_levels=[100, 1000, 10000, 100000],
    ... )
    >>> print(report)
    """
    cost_levels = np.asarray(cost_levels, dtype=np.float64)
    K = len(cost_levels)

    if labels is None:
        if K == 4:
            labels = list(_DEFAULT_LABELS_4)
        else:
            labels = [f"level_{i + 1}" for i in range(K)]
    else:
        labels = list(labels)
        if len(labels) != K:
            raise ValueError(f"len(labels)={len(labels)} != len(cost_levels)={K}")

    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    predictions = list(predictions)
    references = list(references)
    severity_annotations = list(severity_annotations)

    n = len(predictions)
    if n != len(references) or n != len(severity_annotations):
        raise ValueError("predictions, references, and severity_annotations must have same length")

    # Frequency
    correct = sum(1 for p, r in zip(predictions, references, strict=True) if p == r)
    accuracy = correct / n
    error_rate = 1.0 - accuracy

    # Severity profile from errors
    severity_counts = np.zeros(K)
    n_errors = 0
    for p, r, s in zip(predictions, references, severity_annotations, strict=True):
        if p != r:
            if s not in label_to_idx:
                raise ValueError(f"Unknown severity label '{s}'. Expected one of {labels}")
            severity_counts[label_to_idx[s]] += 1
            n_errors += 1

    if n_errors == 0:
        severity_profile = np.zeros(K)
        losses = np.zeros(n_sim)
        measures = {"expected_loss": 0.0, "var": 0.0, "tvar": 0.0}
        ci = (0.0, 0.0)
    else:
        severity_profile = severity_counts / severity_counts.sum()
        losses = simulate_aggregate_loss(
            n_queries,
            error_rate,
            cost_levels,
            severity_profile,
            n_sim,
            seed=seed,
        )
        measures = compute_risk_measures(losses, alpha=alpha)
        ci = bootstrap_ci(losses, statistic="expected_loss", seed=seed)

    # Optional routing analysis
    routing = None
    if routing_threshold is not None and n_errors > 0:
        routing = analyze_routing(
            n_queries=n_queries,
            error_rate=error_rate,
            cost_levels=cost_levels,
            severity_profile=severity_profile,
            retention_threshold=routing_threshold,
            human_review_cost=human_review_cost,
        )

    return SeverityReport(
        cost_levels=cost_levels,
        labels=labels,
        accuracy=accuracy,
        error_rate=error_rate,
        n_samples=n,
        n_errors=n_errors,
        severity_profile=severity_profile,
        severity_counts=severity_counts,
        losses=losses,
        expected_loss=measures["expected_loss"],
        expected_loss_ci=ci,
        var=measures["var"],
        tvar=measures["tvar"],
        alpha=alpha,
        n_queries=n_queries,
        n_sim=n_sim,
        routing=routing,
    )
