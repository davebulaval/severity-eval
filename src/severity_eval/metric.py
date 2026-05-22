"""HuggingFace evaluate-compatible wrapper.

Thin layer over severity_eval.api.evaluate() for integration with
the HuggingFace evaluate library.
"""

from __future__ import annotations

try:
    import datasets
    import evaluate

    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

from severity_eval.api import evaluate as sev_evaluate


def compute_severity_metrics(
    predictions: list,
    references: list,
    severity_annotations: list[str],
    cost_levels: list[int] | list[float] | None = None,
    n_queries: int = 10000,
    n_sim: int = 100000,
    alpha: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compute actuarial metrics and return a flat dict.

    Convenience wrapper that calls severity_eval.evaluate() and
    returns report.to_dict().
    """
    if cost_levels is None:
        raise ValueError("cost_levels is required")
    report = sev_evaluate(
        predictions=predictions,
        references=references,
        severity_annotations=severity_annotations,
        cost_levels=list(cost_levels),
        n_queries=n_queries,
        n_sim=n_sim,
        alpha=alpha,
        seed=seed,
    )
    return report.to_dict()


if _HF_AVAILABLE:

    class CompoundLossMetric(evaluate.Metric):
        """HuggingFace evaluate-compatible compound loss metric."""

        def _info(self) -> evaluate.MetricInfo:
            return evaluate.MetricInfo(
                description=(
                    "Actuarial compound-loss metric combining error frequency "
                    "and severity. Computes E[S], VaR, TVaR via Monte Carlo."
                ),
                citation="",
                inputs_description=(
                    "predictions, references, severity_annotations (each a list of strings of equal length)"
                ),
                features=datasets.Features(
                    {
                        "predictions": datasets.Value("string"),
                        "references": datasets.Value("string"),
                        "severity_annotations": datasets.Value("string"),
                    }
                ),
            )

        def _compute(
            self,
            predictions: list,
            references: list,
            severity_annotations: list[str],
            cost_levels: list[int] | list[float] | None = None,
            n_queries: int = 10000,
            n_sim: int = 100000,
            alpha: float = 0.95,
            seed: int = 42,
        ) -> dict:
            return compute_severity_metrics(
                predictions=predictions,
                references=references,
                severity_annotations=severity_annotations,
                cost_levels=cost_levels,
                n_queries=n_queries,
                n_sim=n_sim,
                alpha=alpha,
                seed=seed,
            )
