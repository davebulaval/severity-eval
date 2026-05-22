"""severity-eval: Actuarial frequency-severity evaluation for language models.

Public surface:

    >>> import severity_eval
    >>> report = severity_eval.evaluate(
    ...     predictions=preds,
    ...     references=refs,
    ...     severity_annotations=labels,
    ...     cost_levels=[100, 1_000, 10_000, 100_000],
    ... )
    >>> print(report)
"""

from __future__ import annotations

from severity_eval.api import SeverityReport, evaluate
from severity_eval.compound_loss import simulate_aggregate_loss
from severity_eval.risk_measures import bootstrap_ci, compute_risk_measures
from severity_eval.routing import RoutingResult, analyze_routing
from severity_eval.ruin import (
    compute_lundberg_R,
    compute_reserve,
    ruin_probability_curve,
    simulate_ruin_probability,
)
from severity_eval.sensitivity import sensitivity_analysis
from severity_eval.taxonomy import (
    SEVERITY_LABELS,
    Taxonomy,
    get_taxonomy,
    list_domains,
    severity_label_to_index,
)

__all__ = [
    "SEVERITY_LABELS",
    "RoutingResult",
    "SeverityReport",
    "Taxonomy",
    "analyze_routing",
    "bootstrap_ci",
    "compute_lundberg_R",
    "compute_reserve",
    "compute_risk_measures",
    "evaluate",
    "get_taxonomy",
    "list_domains",
    "ruin_probability_curve",
    "sensitivity_analysis",
    "severity_label_to_index",
    "simulate_aggregate_loss",
    "simulate_ruin_probability",
]

__version__ = "0.1.0"
