"""Load and annotate FinanceBench dataset with severity levels.

FinanceBench: 150 QA pairs from SEC filings (10-K, 10-Q, 8-K).
Severity is assigned based on two axes:
  1. Answer type: dollar-M/B, dollar-per-share, percentage, ratio, yes/no, text
  2. Metric type: core financials, operational, ratios, descriptive

Justification per SEC SAB 99 materiality framework:
  - critical: core financial statement items (revenue, net income, total assets)
              in absolute dollars — "Big R" restatement territory
  - major:    operational metrics (capex, cash flow, EBITDA, debt, working capital)
              in absolute dollars, or yes/no with numerical backing
  - minor:    ratios, margins, percentages, per-share values
              — "little r" correction territory
  - negligible: descriptive text, company info, filing metadata
"""

from __future__ import annotations

import re

import pandas as pd

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


# --- Metric keywords by severity level ---
# Ordered from most to least severe; first match wins within each level.

CORE_FINANCIALS = [
    "revenue",
    "net income",
    "net earnings",
    "net loss",
    "total assets",
    "total liabilities",
    "total equity",
    "shareholders' equity",
    "stockholders' equity",
    "gross profit",
    "total revenue",
    "cogs",
    "cost of goods sold",
    "cost of sales",
]

OPERATIONAL_METRICS = [
    "ebitda",
    "operating income",
    "operating expense",
    "cash flow",
    "capex",
    "capital expenditure",
    "working capital",
    "free cash flow",
    "free cashflow",
    "debt",
    "long-term debt",
    "short-term debt",
    "inventory",
    "inventories",
    "accounts receivable",
    "accounts payable",
    "current liabilities",
    "current assets",
    "net ar",
    "goodwill",
    "intangible",
    "restructuring",
    "impairment",
    "depreciation",
    "amortization",
    "income tax",
    "tax expense",
    "stock-based compensation",
    "share-based",
    "pension",
    "retiree",
    "derivative",
    "revolving credit",
    "borrowing",
    "capital-intensive",
    "capital intensive",
    "dividend",
    "pp&e",
    "cash & cash equivalents",
    "cash and cash equivalents",
    "legal battle",
    "legal matter",
    "litigation",
    "spinning off",
    "spin-off",
    "spinoff",
    "value at risk",
]

RATIO_KEYWORDS = [
    "ratio",
    "margin",
    "percentage",
    "growth rate",
    "return on",
    "roe",
    "roa",
    "roi",
    "eps",
    "per share",
    "yield",
    "turnover",
    "leverage",
    "liquidity",
    "quick ratio",
    "current ratio",
    "p/e",
    "price-to",
]

DESCRIPTIVE_KEYWORDS = [
    "key agenda",
    "products and services",
    "what are the major",
    "describe",
    "list the",
    "name the",
    "which debt securities",
    "what was the key",
]


def _answer_type(answer: str, question: str) -> str:
    """Classify the answer format.

    Returns one of:
      'dollar_large' — dollar amount likely in millions/billions
      'dollar_small' — dollar amount likely per-share or small
      'percentage'   — answer contains %
      'ratio'        — plain number (0.83, 24.26)
      'yes_no'       — starts with yes/no
      'text'         — free-form text
    """
    a = answer.strip()
    a_lower = a.lower()
    q_lower = question.lower()

    # Yes/No
    if a_lower.startswith("yes") or a_lower.startswith("no"):
        return "yes_no"

    # Dollar amount
    dollar_match = re.search(r"\$([\d,.]+)", a)
    if dollar_match:
        num_str = dollar_match.group(1).replace(",", "")
        try:
            val = float(num_str)
        except ValueError:
            return "text"
        # Answer itself mentions million/billion → definitely large
        if re.search(r"(million|billion)", a_lower):
            return "dollar_large"
        # Small value (<50): likely per-share even if question says "in USD millions"
        if val < 50:
            return "dollar_small"
        # Question mentions millions/billions + value >= 50 → large
        if re.search(r"(million|billion|USD millions|USD billions)", q_lower):
            return "dollar_large"
        # Value > 500 without explicit scale → probably millions
        if val > 500:
            return "dollar_large"
        # Mid-range (50-500) without scale context → small
        return "dollar_small"

    # Percentage
    if "%" in a or "percent" in a_lower:
        return "percentage"

    # Plain number (ratio)
    if re.match(r"^-?[\d,.]+x?$", a.strip()):
        return "ratio"

    return "text"


def _metric_type(question: str) -> str:
    """Classify the financial metric in the question.

    Returns one of:
      'core'         — revenue, net income, total assets, etc.
      'operational'  — capex, cash flow, EBITDA, debt, etc.
      'ratio'        — margins, ratios, percentages, per-share
      'descriptive'  — qualitative, non-numerical

    Priority: descriptive > core > operational > ratio > default.
    Descriptive keywords are checked first to filter out purely
    qualitative questions early.
    """
    q_lower = question.lower()

    for keyword in DESCRIPTIVE_KEYWORDS:
        if keyword in q_lower:
            return "descriptive"

    for keyword in CORE_FINANCIALS:
        if keyword in q_lower:
            return "core"

    for keyword in OPERATIONAL_METRICS:
        if keyword in q_lower:
            return "operational"

    for keyword in RATIO_KEYWORDS:
        if keyword in q_lower:
            return "ratio"

    return "descriptive"


# --- Cross-tabulation: answer_type × metric_type → severity ---

_SEVERITY_MATRIX = {
    # answer_type     metric_type     → severity
    ("dollar_large", "core"): "critical",
    ("dollar_large", "operational"): "major",
    ("dollar_large", "ratio"): "major",
    ("dollar_large", "descriptive"): "major",
    ("dollar_small", "core"): "minor",
    ("dollar_small", "operational"): "minor",
    ("dollar_small", "ratio"): "minor",
    ("dollar_small", "descriptive"): "negligible",
    ("percentage", "core"): "minor",
    ("percentage", "operational"): "minor",
    ("percentage", "ratio"): "minor",
    ("percentage", "descriptive"): "minor",
    ("ratio", "core"): "minor",
    ("ratio", "operational"): "minor",
    ("ratio", "ratio"): "minor",
    ("ratio", "descriptive"): "minor",
    ("yes_no", "core"): "major",
    ("yes_no", "operational"): "major",
    ("yes_no", "ratio"): "minor",
    ("yes_no", "descriptive"): "minor",
    ("text", "core"): "major",
    ("text", "operational"): "minor",
    ("text", "ratio"): "minor",
    ("text", "descriptive"): "negligible",
}


def classify_severity(question: str, answer: str) -> dict:
    """Assign severity based on answer type × metric type.

    Returns dict with severity label and the two axes for traceability.
    """
    at = _answer_type(answer, question)
    mt = _metric_type(question)
    severity = _SEVERITY_MATRIX.get((at, mt), "minor")
    return {
        "severity": severity,
        "answer_type": at,
        "metric_type": mt,
    }


def load_financebench(limit: int | None = None) -> pd.DataFrame:
    """Load FinanceBench and annotate with severity.

    Parameters
    ----------
    limit : int or None
        Max number of instances to load.

    Returns
    -------
    DataFrame with columns:
        id, question, answer, severity, answer_type, metric_type,
        question_type, question_reasoning, company, gics_sector, domain
    """
    if load_dataset is None:
        raise ImportError("Install datasets: pip install datasets")

    ds = load_dataset("PatronusAI/financebench", split="train")

    records = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break

        question = row.get("question", "")
        answer = row.get("answer", "")
        annotation = classify_severity(question, answer)

        records.append(
            {
                "id": row.get("financebench_id", f"financebench_{i}"),
                "question": question,
                "answer": answer,
                "severity": annotation["severity"],
                "answer_type": annotation["answer_type"],
                "metric_type": annotation["metric_type"],
                "question_type": row.get("question_type", ""),
                "question_reasoning": row.get("question_reasoning", ""),
                "company": row.get("company", ""),
                "gics_sector": row.get("gics_sector", ""),
                "domain": "finance",
            }
        )

    return pd.DataFrame(records)
