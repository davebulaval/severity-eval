"""Load and annotate FinQA dataset with severity levels.

FinQA: 8,281 QA pairs from financial reports (SEC filings).
License: CC BY 4.0
Source: https://github.com/czyssrs/FinQA (loaded via GitHub raw URLs)

Note: the ibm/finqa HuggingFace repo relies on a dataset loading script
(finqa.py) that is no longer supported by the ``datasets`` library
(>=2.16).  This loader fetches the JSON files directly from the upstream
GitHub repository instead.

Severity is assigned based on two axes:
  1. Program type: what DSL operations are used (divide → ratio, else arithmetic)
  2. Metric type: what financial concept is in the question

Cross-tabulation following SEC SAB 99 materiality:
  - critical: core financial statement items (revenue, net income, total
              assets) computed with absolute-value arithmetic
  - major:    operational metrics (EBITDA, cash flow, capex, etc.) or
              any unrecognised metric computed with absolute arithmetic
  - minor:    ratios, margins, percentages, growth rates — i.e. any
              program that ultimately divides two numbers, or questions
              about per-unit metrics regardless of ops
  - negligible: comparative yes/no (``greater``) without a numerical
                result, or no program at all
"""

from __future__ import annotations

import json
import re
import urllib.request

import pandas as pd

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------

_GITHUB_BASE = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset"
_SPLITS = {
    "train": f"{_GITHUB_BASE}/train.json",
    "dev": f"{_GITHUB_BASE}/dev.json",
    "test": f"{_GITHUB_BASE}/test.json",
}


# ---------------------------------------------------------------------------
# Keyword lists (ordered longest → shortest within each group so that
# multi-word phrases take priority in substring matching)
# ---------------------------------------------------------------------------

CORE_FINANCIALS = [
    "total revenue",
    "total revenues",
    "net revenue",
    "net revenues",
    "net income",
    "net earnings",
    "net loss",
    "total assets",
    "total liabilities",
    "total equity",
    "shareholders' equity",
    "stockholders' equity",
    "gross profit",
    "gross margin amount",
    "cost of goods sold",
    "cost of sales",
    "cogs",
    "revenue",
]

OPERATIONAL_METRICS = [
    "free cash flow",
    "free cashflow",
    "operating income",
    "operating expense",
    "operating expenses",
    "capital expenditure",
    "capital expenditures",
    "long-term debt",
    "short-term debt",
    "accounts receivable",
    "accounts payable",
    "current liabilities",
    "current assets",
    "cash and cash equivalents",
    "working capital",
    "ebitda",
    "capex",
    "inventory",
    "inventories",
    "goodwill",
    "intangible",
    "cash flow",
    "borrowing",
    "dividend",
    "pp&e",
    "pension",
    "debt",
]

RATIO_KEYWORDS = [
    "growth rate",
    "growth",
    "return on equity",
    "return on assets",
    "price-to-earnings",
    "price-to",
    "percentage",
    "percent",
    "margin",
    "ratio",
    "yield",
    "per share",
    "turnover",
    "roe",
    "roa",
    "roi",
    "eps",
]


def _metric_type(question: str) -> str:
    """Classify the financial concept in the question.

    Returns one of 'core', 'operational', 'ratio', or 'other'.
    Longer / more specific keywords are matched first.
    """
    q = question.lower()

    # Build flat list sorted by keyword length descending so longer phrases win.
    candidates: list[tuple[str, str]] = []
    for kw in CORE_FINANCIALS:
        candidates.append((kw, "core"))
    for kw in OPERATIONAL_METRICS:
        candidates.append((kw, "operational"))
    for kw in RATIO_KEYWORDS:
        candidates.append((kw, "ratio"))

    candidates.sort(key=lambda x: len(x[0]), reverse=True)

    for keyword, mtype in candidates:
        if keyword in q:
            return mtype

    return "other"


def _program_type(program: str) -> str:
    """Classify the DSL program into 'ratio', 'comparison', or 'arithmetic'.

    - 'ratio'      : the last or dominant operation is divide (or the
                     program produces a % result), i.e. numerics divided
    - 'comparison' : only ``greater`` / ``less`` (boolean result)
    - 'arithmetic' : add / subtract / multiply / table_sum / table_avg etc.
    """
    if not program:
        return "comparison"  # no program → no computation → negligible

    ops = re.findall(r"\b([a-z_]+)\s*\(", program)
    if not ops:
        return "arithmetic"

    op_set = set(ops)

    # Pure comparison — no numeric output
    if op_set <= {"greater", "less"}:
        return "comparison"

    # If divide is present, check if it's the final step → produces a ratio/%.
    # We also check if a % literal appears in the args, which signals a ratio.
    if "divide" in op_set:
        # If there are non-comparison, non-divide ops that come AFTER a divide
        # in the program string, the divide is intermediate and the overall
        # result is likely a large number (multiply after divide, etc.).
        last_op = ops[-1]
        if last_op == "divide":
            return "ratio"
        # Divide followed only by itself or table ops → ratio
        non_divide_ops = op_set - {"divide", "greater", "less", "table_max", "table_min", "table_average", "table_sum"}
        if not non_divide_ops:
            return "ratio"
        # Mixed divide + multiply/add → result is likely a big number; treat
        # as arithmetic unless a % literal is in the program.
        if re.search(r"\d+(\.\d+)?%", program):
            return "ratio"
        return "arithmetic"

    return "arithmetic"


# ---------------------------------------------------------------------------
# Severity matrix
# ---------------------------------------------------------------------------

_SEVERITY_MATRIX: dict[tuple[str, str], str] = {
    # program_type   metric_type     → severity
    ("arithmetic", "core"): "critical",
    ("arithmetic", "operational"): "major",
    ("arithmetic", "ratio"): "minor",
    ("arithmetic", "other"): "major",
    ("ratio", "core"): "minor",
    ("ratio", "operational"): "minor",
    ("ratio", "ratio"): "minor",
    ("ratio", "other"): "minor",
    ("comparison", "core"): "negligible",
    ("comparison", "operational"): "negligible",
    ("comparison", "ratio"): "negligible",
    ("comparison", "other"): "negligible",
}


def classify_severity(question: str, program: str) -> dict:
    """Assign severity based on program type × metric type.

    Returns a dict with keys ``severity``, ``program_type``,
    ``metric_type`` for downstream traceability.
    """
    pt = _program_type(program)
    mt = _metric_type(question)
    severity = _SEVERITY_MATRIX.get((pt, mt), "major")
    return {
        "severity": severity,
        "program_type": pt,
        "metric_type": mt,
    }


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_finqa(
    split: str = "test",
    limit: int | None = None,
) -> pd.DataFrame:
    """Load FinQA and annotate with severity.

    Fetches data from the upstream GitHub repository because the official
    ibm/finqa HuggingFace repo uses a deprecated dataset loading script.

    Parameters
    ----------
    split : {'train', 'dev', 'test'}
        Dataset split to load.  Defaults to 'test'.
    limit : int or None
        Maximum number of instances to load.

    Returns
    -------
    DataFrame with columns:
        id, question, answer, severity, program_type, metric_type,
        program, filename, domain
    """
    if split not in _SPLITS:
        raise ValueError(f"split must be one of {list(_SPLITS)}, got {split!r}")

    url = _SPLITS[split]
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download FinQA {split} split from {url}: {exc}") from exc

    records = []
    for i, item in enumerate(raw):
        if limit is not None and i >= limit:
            break

        qa = item.get("qa", {})
        question = qa.get("question", "")
        # 'answer' is the human-readable result; 'exe_ans' is the float
        # execution result.  Use 'answer' for display, fall back to exe_ans.
        answer = qa.get("answer") or str(qa.get("exe_ans", ""))
        program = qa.get("program", "")
        annotation = classify_severity(question, program)

        records.append(
            {
                "id": item.get("id", f"finqa_{split}_{i}"),
                "question": question,
                "answer": str(answer),
                "severity": annotation["severity"],
                "program_type": annotation["program_type"],
                "metric_type": annotation["metric_type"],
                "program": program,
                "filename": item.get("filename", ""),
                "domain": "finance",
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# CLI entry-point for quick sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load FinQA dataset")
    parser.add_argument("--split", default="test", choices=list(_SPLITS))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    df = load_finqa(split=args.split, limit=args.limit)
    print(df.to_string(max_colwidth=80))
    print(f"\nShape: {df.shape}")
    print("\nSeverity distribution:")
    print(df["severity"].value_counts())
