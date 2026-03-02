"""severity-eval: Actuarial compound-loss metrics for AI evaluation."""

from severity_eval.api import SeverityReport, evaluate
from severity_eval.compound_loss import simulate_aggregate_loss
from severity_eval.risk_measures import bootstrap_ci, compute_risk_measures
from severity_eval.routing import analyze_routing
from severity_eval.ruin import (
    compute_lundberg_R,
    compute_reserve,
    ruin_probability_curve,
    simulate_ruin_probability,
)
from severity_eval.sensitivity import sensitivity_analysis
from severity_eval.taxonomy import DOMAINS, get_taxonomy, list_domains, register_taxonomy
from severity_eval.validation import validate_severity_profile

__version__ = "0.1.0"

__all__ = [
    "DOMAINS",
    "SeverityReport",
    "analyze_routing",
    "bootstrap_ci",
    "compute_lundberg_R",
    "compute_reserve",
    "compute_risk_measures",
    "evaluate",
    "get_taxonomy",
    "list_domains",
    "register_taxonomy",
    "ruin_probability_curve",
    "sensitivity_analysis",
    "simulate_aggregate_loss",
    "simulate_ruin_probability",
    "validate_severity_profile",
]
