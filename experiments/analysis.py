"""Statistical analysis: compute actuarial metrics, test hypotheses, generate tables.

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
from severity_eval.risk_measures import (
    bootstrap_ci,
    compute_risk_measures,
    wilson_score_interval,
)
from severity_eval.routing import analyze_routing
from severity_eval.taxonomy import get_taxonomy, severity_label_to_index

# ---------------------------------------------------------------------------
# Domain normalisation
# ---------------------------------------------------------------------------

# Map raw `domain` fields from dataset loaders to the canonical taxonomy
# domain used for cost-vector lookup. Datasets that share a regulatory
# regime (e.g. legal NLI, legal simplification) collapse to the same
# severity scale.
_DOMAIN_TO_TAXONOMY = {
    "finance": "finance",
    "medical": "medical",
    "legal": "legal",
    "legal_nli": "legal",
    "legal_simplification": "legal",
    "insurance": "insurance",
}


def _canonical_domain(domain: str) -> str:
    """Normalise a dataset domain string for taxonomy lookup."""
    return _DOMAIN_TO_TAXONOMY.get(domain, domain)


# Datasets retained for the paper after the severity-annotation audit.
# medqa, medcalc, and rag_insurance were dropped (keyword-based severity
# that inter-annotator agreement could not defend) -- see the Limitations
# section. Their JSON files may still sit in results/ from earlier smoke
# runs, so load_results filters them out by default to keep the per-domain
# aggregates (Kendall tau, variance decomposition, routing) clean. Pass
# include_dropped=True to re-include them for replication of the dropped
# benchmarks.
RETAINED_DATASETS = (
    "financebench",
    "finqa",
    "tatqa",
    "headqa",
    "medmcqa",
    "ddi",
    "cuad",
    "maud",
    "contractnli",
    "judgebert",
    "privacyqa",
)
_DROPPED_DATASETS = ("medqa", "medcalc", "rag_insurance")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_results(results_dir: Path, include_dropped: bool = False) -> pd.DataFrame:
    """Load all evaluation results into a single DataFrame.

    Filename convention: ``<dataset>_<model>[_<prompt_style>].json``. The
    dataset and prompt style are extracted from the file name; the model
    field is read from each record (and cross-checked against the file).

    By default only the eight retained datasets are loaded; the three
    dropped datasets (medqa, medcalc, rag_insurance) are skipped so stale
    smoke-run JSONs cannot contaminate the per-domain aggregates. Set
    ``include_dropped=True`` to load everything.
    """
    dfs = []
    for f in sorted(results_dir.glob("*.json")):
        df = pd.read_json(f, orient="records")
        if df.empty:
            continue
        stem = f.stem
        # Extract prompt style suffix
        if stem.endswith("_standard"):
            prompt_style = "standard"
            stem = stem[: -len("_standard")]
        else:
            prompt_style = "original"
        # The dataset is the part before the first underscore, but a few
        # dataset names contain underscores (rag_insurance). Match against
        # the known dataset list to disambiguate.
        known_datasets = (
            "financebench",
            "finqa",
            "tatqa",
            "medcalc",
            "medqa",
            "headqa",
            "medmcqa",
            "ddi",
            "cuad",
            "maud",
            "contractnli",
            "rag_insurance",
            "judgebert",
            "privacyqa",
        )
        ds_match = next((d for d in known_datasets if stem.startswith(d + "_")), None)
        if ds_match is None:
            # Fall back to the first segment.
            ds_match = stem.split("_", 1)[0]
        if not include_dropped and ds_match in _DROPPED_DATASETS:
            continue
        df["dataset"] = ds_match
        df["prompt_style"] = prompt_style
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No result files found in {results_dir}")
    full = pd.concat(dfs, ignore_index=True)
    full["taxonomy_domain"] = (
        full["domain"].map(_canonical_domain).fillna(full["domain"])
    )
    return full


# ---------------------------------------------------------------------------
# Per-(model, dataset) actuarial metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    df: pd.DataFrame,
    n_queries: int = 10_000,
    n_sim: int = 20_000,
    alpha: float = 0.95,
    seed: int = 42,
    prompt_style: str | None = "original",
) -> pd.DataFrame:
    """Compute actuarial metrics for every (model, dataset) pair.

    Parameters
    ----------
    df : DataFrame
        Loaded by :func:`load_results`.
    prompt_style : str or None
        If provided, restrict to one prompt style. ``None`` aggregates all.
    """
    if prompt_style is not None and "prompt_style" in df.columns:
        df = df[df["prompt_style"] == prompt_style].copy()

    records = []
    for (model, dataset), group in df.groupby(["model", "dataset"]):
        n = len(group)
        domain_raw = group["domain"].iloc[0]
        tax_domain = _canonical_domain(domain_raw)
        try:
            taxonomy = get_taxonomy(tax_domain)
        except KeyError:
            # Skip datasets without a defined taxonomy.
            continue

        correct = int(group["correct"].sum())
        accuracy = correct / n if n else 0.0
        error_rate = 1.0 - accuracy

        counts = np.zeros(len(taxonomy.cost_levels))
        for sev in group.loc[~group["correct"], "severity"]:
            try:
                idx = severity_label_to_index(str(sev))
            except ValueError:
                continue
            counts[idx] += 1

        if counts.sum() == 0:
            # No errors observed -> degenerate severity profile, all
            # loss-side CIs collapse to 0. Accuracy CI is still
            # well-defined and reflects n.
            acc_lo, acc_hi = wilson_score_interval(correct, n)
            records.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "domain": domain_raw,
                    "taxonomy_domain": tax_domain,
                    "n": n,
                    "n_errors": 0,
                    "accuracy": accuracy,
                    "accuracy_ci_lo": acc_lo,
                    "accuracy_ci_hi": acc_hi,
                    "error_rate": error_rate,
                    "severity_profile": [0.0] * len(taxonomy.cost_levels),
                    "mu_X": 0.0,
                    "expected_loss": 0.0,
                    "expected_loss_ci_lo": 0.0,
                    "expected_loss_ci_hi": 0.0,
                    "var": 0.0,
                    "var_ci_lo": 0.0,
                    "var_ci_hi": 0.0,
                    "tvar": 0.0,
                    "tvar_ci_lo": 0.0,
                    "tvar_ci_hi": 0.0,
                }
            )
            continue

        pi = counts / counts.sum()
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
        var_lo, var_hi = bootstrap_ci(S, statistic="var", alpha=alpha, seed=seed)
        tvar_lo, tvar_hi = bootstrap_ci(S, statistic="tvar", alpha=alpha, seed=seed)
        # Wilson 95% CI on accuracy : robust at p_hat=0 or p_hat=1 unlike
        # the normal approximation. Critical because several (model,
        # dataset) cells sit at the extremes (e.g. gpt-oss-20b on maud at
        # ~0.99).
        acc_lo, acc_hi = wilson_score_interval(correct, n)
        mu_X = float((taxonomy.cost_levels * pi).sum())

        records.append(
            {
                "model": model,
                "dataset": dataset,
                "domain": domain_raw,
                "taxonomy_domain": tax_domain,
                "n": n,
                "n_errors": int(counts.sum()),
                "accuracy": accuracy,
                "accuracy_ci_lo": acc_lo,
                "accuracy_ci_hi": acc_hi,
                "error_rate": error_rate,
                "severity_profile": pi.tolist(),
                "mu_X": mu_X,
                "expected_loss": measures["expected_loss"],
                "expected_loss_ci_lo": ci_lo,
                "expected_loss_ci_hi": ci_hi,
                "var": measures["var"],
                "var_ci_lo": var_lo,
                "var_ci_hi": var_hi,
                "tvar": measures["tvar"],
                "tvar_ci_lo": tvar_lo,
                "tvar_ci_hi": tvar_hi,
            }
        )

    metrics = pd.DataFrame(records)
    if metrics.empty:
        return metrics
    return metrics.sort_values(["taxonomy_domain", "dataset", "expected_loss"])


def aggregate_per_domain(metrics: pd.DataFrame) -> pd.DataFrame:
    """Average risk measures per (model, taxonomy_domain).

    Multiple datasets share the same taxonomy in some domains (finance
    has FinanceBench, FinQA, TAT-QA, etc.). We average within a domain
    weighted by ``n_errors`` so domains with more data dominate.
    """
    if metrics.empty:
        return metrics
    rows = []
    for (model, dom), group in metrics.groupby(["model", "taxonomy_domain"]):
        n_total = int(group["n"].sum())
        n_err = int(group["n_errors"].sum())
        accuracy = float(group["accuracy"].mean())
        # n-weighted aggregate
        if n_err > 0:
            weights = group["n_errors"].to_numpy()
            profiles = np.array(group["severity_profile"].tolist())
            mean_pi = (profiles * weights[:, None]).sum(0) / weights.sum()
        else:
            mean_pi = np.zeros(len(group["severity_profile"].iloc[0]))
        rows.append(
            {
                "model": model,
                "taxonomy_domain": dom,
                "n_datasets": int(group["dataset"].nunique()),
                "n_total": n_total,
                "accuracy": accuracy,
                "error_rate": 1 - accuracy,
                "expected_loss": float(group["expected_loss"].mean()),
                "var": float(group["var"].mean()),
                "tvar": float(group["tvar"].mean()),
                "severity_profile": mean_pi.tolist(),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Hypothesis tests
# ---------------------------------------------------------------------------


def test_h1_ranking_divergence(metrics: pd.DataFrame, weighted: bool = False) -> dict:
    """H1: Kendall tau between accuracy and E[S] rankings per domain.

    The aggregation across datasets within a domain affects the per-model
    summary used to compute the rank correlation. We expose both choices:

    - ``weighted=False`` (default, reported in the paper body): unweighted
      mean of per-(model, dataset) accuracy and expected loss. This is the
      pandas ``groupby("model").mean()`` convention.
    - ``weighted=True`` (reported in Appendix G as a sensitivity check):
      sample-weighted mean where each dataset contributes proportionally
      to its corpus size ``n``. This matches the per-output aggregation
      used in the Figure 1 slopegraph.

    Both aggregations are reported because the choice changes the point
    estimate of tau (e.g., finance: 0.58 unweighted vs 0.67 weighted) but
    not the qualitative cross-domain ordering.
    """
    out = {}
    n_divergent = 0
    for dom in metrics["taxonomy_domain"].unique():
        df = metrics[metrics["taxonomy_domain"] == dom].copy()
        if df["model"].nunique() < 3:
            continue
        if weighted:
            # Sample-weighted mean within domain.
            df["_acc_n"] = df["accuracy"] * df["n"]
            df["_es_n"] = df["expected_loss"] * df["n"]
            grp = df.groupby("model").agg(
                _acc_n=("_acc_n", "sum"),
                _es_n=("_es_n", "sum"),
                _n=("n", "sum"),
            )
            grp["accuracy"] = grp["_acc_n"] / grp["_n"]
            grp["expected_loss"] = grp["_es_n"] / grp["_n"]
            agg = grp.reset_index()[["model", "accuracy", "expected_loss"]]
        else:
            agg = (
                df.groupby("model")
                .agg(
                    accuracy=("accuracy", "mean"),
                    expected_loss=("expected_loss", "mean"),
                )
                .reset_index()
            )
        acc_rank = agg["accuracy"].rank(ascending=False)
        loss_rank = agg["expected_loss"].rank(ascending=True)
        tau, p = kendalltau(acc_rank, loss_rank)
        divergent = bool(tau < 0.5 and p < 0.05)
        n_divergent += int(divergent)
        out[dom] = {
            "tau": float(tau),
            "p_value": float(p),
            "divergent": divergent,
            "n_models": int(len(agg)),
            "aggregation": "weighted" if weighted else "unweighted",
        }
    return {
        "per_domain": out,
        "n_divergent": n_divergent,
        "hypothesis_supported": n_divergent >= 3,
    }


def test_h3_inversions(metrics: pd.DataFrame) -> dict:
    """H3: pairs (A, B) where accuracy(A) > accuracy(B) but E[S](A) > E[S](B)."""
    inversions = []
    for (dom, ds), group in metrics.groupby(["taxonomy_domain", "dataset"]):
        rows = group.to_dict("records")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                if (
                    a["accuracy"] > b["accuracy"]
                    and a["expected_loss"] > b["expected_loss"]
                ):
                    inversions.append(
                        {
                            "domain": dom,
                            "dataset": ds,
                            "better_acc": a["model"],
                            "worse_acc": b["model"],
                            "acc_gap": a["accuracy"] - b["accuracy"],
                            "loss_gap": a["expected_loss"] - b["expected_loss"],
                        }
                    )
                elif (
                    b["accuracy"] > a["accuracy"]
                    and b["expected_loss"] > a["expected_loss"]
                ):
                    inversions.append(
                        {
                            "domain": dom,
                            "dataset": ds,
                            "better_acc": b["model"],
                            "worse_acc": a["model"],
                            "acc_gap": b["accuracy"] - a["accuracy"],
                            "loss_gap": b["expected_loss"] - a["expected_loss"],
                        }
                    )
    return {
        "n_inversions": len(inversions),
        "inversions": inversions,
        "hypothesis_supported": len(inversions) > 0,
    }


def test_h4_routing(
    metrics: pd.DataFrame, n_queries: int = 10_000, human_review_cost: float = 50.0
) -> dict:
    """H4: routing benefit increases with cost-vector heaviness."""
    out = {}
    for dom in metrics["taxonomy_domain"].unique():
        try:
            taxonomy = get_taxonomy(dom)
        except KeyError:
            continue
        df = metrics[metrics["taxonomy_domain"] == dom]
        reductions, rhos = [], []
        for _, row in df.iterrows():
            pi = np.asarray(row["severity_profile"], dtype=float)
            if row["error_rate"] == 0 or pi.sum() == 0:
                continue
            res = analyze_routing(
                n_queries=n_queries,
                error_rate=row["error_rate"],
                cost_levels=taxonomy.cost_levels,
                severity_profile=pi,
                retention_threshold=taxonomy.cost_levels[2],
                human_review_cost=human_review_cost,
            )
            reductions.append(res.cost_reduction_pct)
            rhos.append(res.routing_ratio)
        cv = float(np.std(taxonomy.cost_levels) / np.mean(taxonomy.cost_levels))
        out[dom] = {
            "cost_cv": cv,
            "mean_reduction_pct": float(np.mean(reductions)) if reductions else 0.0,
            "median_routing_ratio": float(np.median(rhos)) if rhos else 0.0,
            "n_models": len(reductions),
        }
    return out


def test_h2_variance_decomposition(
    metrics: pd.DataFrame,
    n_queries: int = 10_000,
    n_sim: int = 10_000,
) -> dict:
    """H2: which factor drives more variance in E[S]: severity or frequency?"""
    out = {}
    for dom in metrics["taxonomy_domain"].unique():
        try:
            taxonomy = get_taxonomy(dom)
        except KeyError:
            continue
        df = metrics[metrics["taxonomy_domain"] == dom]
        if df["model"].nunique() < 3:
            continue
        # Use one (representative) row per model.
        agg = (
            df.groupby("model")
            .agg(
                accuracy=("accuracy", "mean"),
                error_rate=("error_rate", "mean"),
                expected_loss=("expected_loss", "mean"),
                severity_profile=("severity_profile", "first"),
            )
            .reset_index()
        )
        baseline_es = agg["expected_loss"].to_numpy()
        if baseline_es.size < 2:
            continue
        # 1) Fix p to mean, vary pi
        mean_p = float(agg["error_rate"].mean())
        es_pi_only = []
        for _, row in agg.iterrows():
            pi = np.asarray(row["severity_profile"], dtype=float)
            if pi.sum() == 0:
                es_pi_only.append(0.0)
                continue
            S = simulate_aggregate_loss(
                n_queries, mean_p, taxonomy.cost_levels, pi, n_sim, seed=42
            )
            es_pi_only.append(float(S.mean()))
        # 2) Fix pi to (n-weighted) mean, vary p
        profiles = np.array(agg["severity_profile"].tolist())
        mean_pi = profiles.mean(0)
        if mean_pi.sum() > 0:
            mean_pi = mean_pi / mean_pi.sum()
        es_p_only = []
        for _, row in agg.iterrows():
            if mean_pi.sum() == 0:
                es_p_only.append(0.0)
                continue
            S = simulate_aggregate_loss(
                n_queries,
                float(row["error_rate"]),
                taxonomy.cost_levels,
                mean_pi,
                n_sim,
                seed=42,
            )
            es_p_only.append(float(S.mean()))

        var_base = float(np.var(baseline_es))
        var_pi = float(np.var(es_pi_only))
        var_p = float(np.var(es_p_only))
        out[dom] = {
            "var_baseline": var_base,
            "var_severity_only": var_pi,
            "var_frequency_only": var_p,
            "severity_share": var_pi / (var_pi + var_p)
            if (var_pi + var_p) > 0
            else 0.0,
            "severity_dominates": var_pi > var_p,
        }
    n_dominates = sum(1 for v in out.values() if v["severity_dominates"])
    return {
        "per_domain": out,
        "hypothesis_supported": n_dominates > len(out) / 2 if out else False,
    }


def test_h5_robustness(
    metrics: pd.DataFrame,
    n_queries: int = 10_000,
    n_sim: int = 10_000,
    perturbation: float = 0.20,
) -> dict:
    """H5: Spearman correlation of E[S] rankings under cost perturbation."""
    from severity_eval.sensitivity import sensitivity_analysis

    out = {}
    for dom in metrics["taxonomy_domain"].unique():
        try:
            taxonomy = get_taxonomy(dom)
        except KeyError:
            continue
        df = metrics[metrics["taxonomy_domain"] == dom]
        if df["model"].nunique() < 2:
            continue
        agg = (
            df.groupby("model")
            .agg(
                error_rate=("error_rate", "mean"),
                severity_profile=("severity_profile", "first"),
            )
            .reset_index()
        )
        error_rates = {
            row["model"]: float(row["error_rate"]) for _, row in agg.iterrows()
        }
        profiles = {}
        for _, row in agg.iterrows():
            pi = np.asarray(row["severity_profile"], dtype=float)
            if pi.sum() == 0:
                pi = np.array([1.0, 0.0, 0.0, 0.0])
            profiles[row["model"]] = pi
        sa = sensitivity_analysis(
            n_queries=n_queries,
            error_rates=error_rates,
            cost_levels=taxonomy.cost_levels,
            severity_profiles=profiles,
            perturbation=perturbation,
            n_sim=n_sim,
            seed=42,
        )
        min_rho = float(sa["min_spearman"])
        mean_rho = float(np.mean(sa["spearman_correlations"]))
        out[dom] = {
            "min_spearman": min_rho,
            "mean_spearman": mean_rho,
            "stable": min_rho >= 0.8,
        }
    n_stable = sum(1 for v in out.values() if v["stable"])
    return {
        "per_domain": out,
        "hypothesis_supported": n_stable == len(out) and len(out) > 0,
    }


# ---------------------------------------------------------------------------
# LaTeX export
# ---------------------------------------------------------------------------


def generate_latex_main_table(metrics: pd.DataFrame, output_path: Path):
    """Write the main results table (model x dataset) as LaTeX."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering\footnotesize",
        r"\caption{Accuracy and actuarial risk measures per (model, dataset).",
        r"Costs are in dollars. $\hat{p}$ is the empirical error rate, $\widehat{E[S]}$ the",
        r"expected aggregate loss for $n{=}10{,}000$ queries, with 95\% bootstrap CI.",
        r"Lower $\widehat{E[S]}$ is better.}",
        r"\label{tab:main-results}",
        r"\begin{tabular}{lllrrrrrr}",
        r"\toprule",
        r"Domain & Dataset & Model & Acc.~(\%) & $\hat{p}$ & $\widehat{E[S]}$ & CI 95\% & VaR$_{95}$ & TVaR$_{95}$ \\",
        r"\midrule",
    ]
    for tax_dom in sorted(metrics["taxonomy_domain"].unique()):
        sub = metrics[metrics["taxonomy_domain"] == tax_dom]
        for dataset in sorted(sub["dataset"].unique()):
            ds_df = sub[sub["dataset"] == dataset].sort_values("expected_loss")
            first = True
            for _, row in ds_df.iterrows():
                dom_str = tax_dom.capitalize() if first else ""
                ds_str = dataset.replace("_", r"\_") if first else ""
                lines.append(
                    f"  {dom_str} & {ds_str} & {row['model']} & "
                    f"{row['accuracy'] * 100:.1f} & "
                    f"{row['error_rate']:.3f} & "
                    f"\\${row['expected_loss']:,.0f} & "
                    f"[\\${row['expected_loss_ci_lo']:,.0f}, \\${row['expected_loss_ci_hi']:,.0f}] & "
                    f"\\${row['var']:,.0f} & "
                    f"\\${row['tvar']:,.0f} \\\\"
                )
                first = False
            lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")
    lines += [r"\end{tabular}", r"\end{table*}"]
    output_path.write_text("\n".join(lines))


def generate_latex_h1_table(h1: dict, output_path: Path):
    rows = [
        r"\begin{table}[t]\centering\small",
        r"\caption{H1 -- Kendall $\tau$ between accuracy and $E[S]$ rankings per",
        r"taxonomy domain.}\label{tab:h1}",
        r"\begin{tabular}{lrrrl}",
        r"\toprule",
        r"Domain & \#Models & $\tau$ & $p$ & Divergent? \\",
        r"\midrule",
    ]
    for dom, r in sorted(h1["per_domain"].items()):
        rows.append(
            f"  {dom.capitalize()} & {r['n_models']} & {r['tau']:.3f} & {r['p_value']:.3f} & "
            f"{'Yes' if r['divergent'] else 'No'} \\\\"
        )
    rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    output_path.write_text("\n".join(rows))


def generate_latex_h4_table(h4: dict, output_path: Path):
    rows = [
        r"\begin{table}[t]\centering\small",
        r"\caption{H4 -- HITL routing benefit by domain (retention threshold $d{=}c_3$,",
        r"human review cost $h{=}\$50$).}\label{tab:h4}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Domain & CV($c$) & \#Models & Median $\rho$ & Mean reduction (\%) \\",
        r"\midrule",
    ]
    for dom, r in sorted(h4.items()):
        rows.append(
            f"  {dom.capitalize()} & {r['cost_cv']:.2f} & {r['n_models']} & "
            f"{r['median_routing_ratio']:.2f} & {r['mean_reduction_pct']:.1f} \\\\"
        )
    rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    output_path.write_text("\n".join(rows))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def log_to_wandb(
    metrics: pd.DataFrame,
    domain_metrics: pd.DataFrame,
    h1: dict,
    h2: dict,
    h3: dict,
    h4: dict,
    h5: dict,
    project: str = "severity-eval",
    run_name: str | None = None,
) -> None:
    """Push the aggregated analysis to W&B as a single synthesis run.

    Complements the per-(dataset, model) evaluation runs already logged by
    `evaluate_models.py` with one consolidated "analysis" run containing
    summary statistics, the full metrics table, and the RQ1-RQ5 results.
    """
    try:
        import wandb
    except ImportError:
        print("[WARN] wandb not installed; skipping --wandb")
        return

    if run_name is None:
        from datetime import datetime

        run_name = f"analysis-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    run = wandb.init(
        project=project, name=run_name, job_type="analysis", tags=["analysis"]
    )

    # Per-(model, dataset) table
    cols = [
        "model",
        "dataset",
        "domain",
        "taxonomy_domain",
        "n",
        "n_errors",
        "accuracy",
        "error_rate",
        "mu_X",
        "expected_loss",
        "expected_loss_ci_lo",
        "expected_loss_ci_hi",
        "var",
        "tvar",
    ]
    metric_table = wandb.Table(
        columns=cols,
        data=metrics[cols].astype(object).values.tolist(),
    )
    wandb.log({"metrics_per_model_dataset": metric_table})

    # Per-domain aggregate
    dom_cols = [c for c in domain_metrics.columns if c != "severity_profile"]
    if not domain_metrics.empty:
        dom_table = wandb.Table(
            columns=dom_cols,
            data=domain_metrics[dom_cols].astype(object).values.tolist(),
        )
        wandb.log({"metrics_per_domain": dom_table})

    # Summary scalars
    summary = {
        "n_predictions": int(metrics["n"].sum()),
        "n_models": int(metrics["model"].nunique()),
        "n_datasets": int(metrics["dataset"].nunique()),
        "mean_accuracy": float(metrics["accuracy"].mean()),
        "median_accuracy": float(metrics["accuracy"].median()),
        "h1_n_divergent_domains": h1.get("n_divergent", 0),
        "h1_supported": int(h1.get("hypothesis_supported", False)),
        "h2_supported": int(h2.get("hypothesis_supported", False)),
        "h3_n_inversions": h3.get("n_inversions", 0),
        "h5_supported": int(h5.get("hypothesis_supported", False)),
    }
    for dom, r in h1.get("per_domain", {}).items():
        summary[f"h1_tau_{dom}"] = r["tau"]
        summary[f"h1_pvalue_{dom}"] = r["p_value"]
    for dom, r in h4.items():
        summary[f"h4_mean_reduction_{dom}"] = r["mean_reduction_pct"]
        summary[f"h4_cost_cv_{dom}"] = r["cost_cv"]
    for dom, r in h5.get("per_domain", {}).items():
        summary[f"h5_min_spearman_{dom}"] = r["min_spearman"]

    wandb.summary.update(summary)

    # Inversions table (RQ3)
    if h3.get("inversions"):
        inv_cols = [
            "domain",
            "dataset",
            "better_acc",
            "worse_acc",
            "acc_gap",
            "loss_gap",
        ]
        inv_table = wandb.Table(
            columns=inv_cols,
            data=[[i[c] for c in inv_cols] for i in h3["inversions"][:200]],
        )
        wandb.log({"rq3_inversions": inv_table})

    run.finish()
    print(f"[OK] W&B analysis run logged: {project}/{run_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyse severity-aware evaluation results"
    )
    parser.add_argument("--results-dir", type=Path, default=Path("experiments/results"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument(
        "--prompt-style",
        default=None,
        choices=["original", "standard", None],
        help="Filter by prompt style; default uses all",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log the aggregated analysis to W&B as a synthesis run",
    )
    parser.add_argument(
        "--wandb-project",
        default="severity-eval",
        help="W&B project name (default: severity-eval)",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading results from", args.results_dir)
    df = load_results(args.results_dir)
    print(
        f"  {len(df)} predictions, {df['model'].nunique()} models, {df['dataset'].nunique()} datasets"
    )

    print("\nComputing actuarial metrics ...")
    metrics = compute_metrics(df, prompt_style=args.prompt_style)
    if metrics.empty:
        print("No metrics could be computed (no taxonomy match?)")
        return
    print(
        metrics[
            ["model", "dataset", "accuracy", "expected_loss", "var", "tvar"]
        ].to_string()
    )
    metrics.to_csv(args.output / "metrics.csv", index=False)
    metrics.to_json(args.output / "metrics.json", orient="records", indent=2)

    # Per-domain aggregate
    domain_metrics = aggregate_per_domain(metrics)
    domain_metrics.to_csv(args.output / "metrics_by_domain.csv", index=False)

    # Hypotheses
    print("\nH1: Ranking divergence (Kendall tau)")
    h1 = test_h1_ranking_divergence(metrics)
    for dom, r in h1["per_domain"].items():
        status = "DIVERGENT" if r["divergent"] else "aligned"
        print(f"  {dom}: tau={r['tau']:.3f}, p={r['p_value']:.4f} ({status})")
    print(f"  => H1 supported: {h1['hypothesis_supported']}")

    print("\nH2: Variance decomposition")
    h2 = test_h2_variance_decomposition(metrics)
    for dom, r in h2["per_domain"].items():
        print(
            f"  {dom}: severity_share={r['severity_share']:.2f} (dominates={r['severity_dominates']})"
        )
    print(f"  => H2 supported: {h2['hypothesis_supported']}")

    print("\nH3: Ranking inversions")
    h3 = test_h3_inversions(metrics)
    print(f"  Found {h3['n_inversions']} inversions")
    for inv in h3["inversions"][:5]:
        print(
            f"    {inv['dataset']}: {inv['better_acc']} (+{inv['acc_gap']:.3f} acc) "
            f"costs ${inv['loss_gap']:,.0f} more than {inv['worse_acc']}"
        )

    print("\nH4: Routing benefit by domain (HITL)")
    h4 = test_h4_routing(metrics)
    for dom, r in h4.items():
        print(
            f"  {dom}: CV={r['cost_cv']:.2f}, mean reduction={r['mean_reduction_pct']:.1f}%"
        )

    print("\nH5: Robustness to cost perturbation (+/-20%)")
    h5 = test_h5_robustness(metrics)
    for dom, r in h5["per_domain"].items():
        print(
            f"  {dom}: min Spearman={r['min_spearman']:.3f}, mean={r['mean_spearman']:.3f}"
        )
    print(f"  => H5 supported: {h5['hypothesis_supported']}")

    # Persist hypothesis tests
    with open(args.output / "hypothesis_tests.json", "w") as f:
        json.dump(
            {"h1": h1, "h2": h2, "h3": h3, "h4": h4, "h5": h5},
            f,
            indent=2,
            default=str,
        )

    # LaTeX tables
    generate_latex_main_table(metrics, args.output / "table_main.tex")
    generate_latex_h1_table(h1, args.output / "table_h1.tex")
    generate_latex_h4_table(h4, args.output / "table_h4.tex")

    # W&B logging (optional)
    if args.wandb:
        log_to_wandb(
            metrics, domain_metrics, h1, h2, h3, h4, h5, project=args.wandb_project
        )

    print(f"\nAll outputs saved to {args.output}/")


if __name__ == "__main__":
    main()
