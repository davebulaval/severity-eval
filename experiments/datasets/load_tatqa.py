"""Load and annotate TAT-QA dataset with severity levels.

TAT-QA: 16,552 QA pairs from financial reports.
License: CC BY 4.0
Source: next-tat/TAT-QA on HuggingFace

The dataset lives in three JSON files; the train split cannot be loaded
via the ``datasets`` streaming API (schema inconsistency between splits),
so this loader fetches the JSON files directly from HuggingFace.

Each top-level item represents a financial document passage and contains
a list of ``questions``.  This loader flattens all questions across all
document passages into a single DataFrame row per QA pair.

Severity is assigned based on two axes following SEC SAB 99 materiality:
  1. scale: billion > million > thousand > percent > (empty string)
  2. answer_type: arithmetic > count > multi-span > span

Cross-tabulation:
  - critical: scale="billion" + arithmetic, or scale="million" + arithmetic
              AND question references a core financial item
  - major:    scale="million" + arithmetic/count (non-core), or
              scale="billion" + non-arithmetic
  - minor:    scale="thousand", scale="percent", or answer_type="span"
              with numeric content, or multi-span
  - negligible: answer_type="span"/"multi-span" with text content,
                scale="" (no scale)
"""

from __future__ import annotations

import json
import re
import urllib.request

import pandas as pd

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------

_HF_BASE = "https://huggingface.co/datasets/next-tat/TAT-QA/resolve/main"
_SPLITS = {
    "train": f"{_HF_BASE}/tatqa_dataset_train.json",
    "dev": f"{_HF_BASE}/tatqa_dataset_dev.json",
    # Use test_gold which includes ground-truth answers (tatqa_dataset_test.json
    # is the blind test set without answers)
    "test": f"{_HF_BASE}/tatqa_dataset_test_gold.json",
}


# ---------------------------------------------------------------------------
# Keyword lists for core financial items
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
    "cost of goods sold",
    "cost of sales",
    "revenue",
]


def _is_core_financial(question: str) -> bool:
    """Return True if the question targets a core financial statement item."""
    q = question.lower()
    # Sort by length so longer / more specific phrases match first.
    for kw in sorted(CORE_FINANCIALS, key=len, reverse=True):
        if kw in q:
            return True
    return False


def _answer_is_numeric(answer: str | list) -> bool:
    """Return True if the answer contains a numeric value."""
    if isinstance(answer, list):
        answer = " ".join(str(a) for a in answer)
    return bool(re.search(r"\d", str(answer)))


# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------


def classify_severity(
    question: str,
    answer_type: str,
    scale: str,
    answer: str | list,
) -> dict:
    """Assign severity based on scale × answer_type × financial concept.

    Parameters
    ----------
    question : str
    answer_type : {'arithmetic', 'count', 'multi-span', 'span'}
    scale : {'billion', 'million', 'thousand', 'percent', ''}
    answer : str or list

    Returns
    -------
    dict with keys ``severity``, ``scale``, ``answer_type``.
    """
    scale = (scale or "").lower().strip()
    answer_type = (answer_type or "").lower().strip()
    is_core = _is_core_financial(question)
    is_numeric = _answer_is_numeric(answer)

    # ---- critical ----
    if (scale == "billion" and answer_type == "arithmetic") or (scale == "million" and answer_type == "arithmetic" and is_core):
        severity = "critical"

    # ---- major ----
    elif scale == "billion":
        # billion + non-arithmetic → still very large figure
        severity = "major"
    elif (scale == "million" and answer_type in ("arithmetic", "count")) or (scale == "million" and is_core):
        severity = "major"

    # ---- minor ----
    elif scale in ("thousand", "percent"):
        severity = "minor"
    elif scale == "million":
        # million + span → extractive, lower risk
        severity = "minor"
    elif answer_type == "arithmetic":
        # arithmetic without a scale → result could be a ratio or small figure
        severity = "minor"
    elif (answer_type in ("count", "multi-span") and is_numeric) or (answer_type == "span" and is_numeric):
        severity = "minor"

    # ---- negligible ----
    else:
        severity = "negligible"

    return {
        "severity": severity,
        "scale": scale,
        "answer_type": answer_type,
    }


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_tatqa(
    split: str = "test",
    limit: int | None = None,
) -> pd.DataFrame:
    """Load TAT-QA and annotate with severity.

    Fetches JSON data directly from HuggingFace because the ``datasets``
    streaming API cannot handle TAT-QA's schema inconsistency in the train
    split (mixed list / non-list values in the ``questions`` column).

    Parameters
    ----------
    split : {'train', 'dev', 'test'}
        Dataset split to load.  'test' uses the gold-annotated test file.
        Defaults to 'test'.
    limit : int or None
        Maximum number of QA pairs to return (across all document passages).

    Returns
    -------
    DataFrame with columns:
        id, question, answer, severity, scale, answer_type, answer_from,
        derivation, domain
    """
    if split not in _SPLITS:
        raise ValueError(f"split must be one of {list(_SPLITS)}, got {split!r}")

    url = _SPLITS[split]
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download TAT-QA {split} split from {url}: {exc}") from exc

    records = []
    for passage in raw:
        if limit is not None and len(records) >= limit:
            break

        questions = passage.get("questions") or []
        for q in questions:
            if limit is not None and len(records) >= limit:
                break

            question_text = q.get("question", "")
            answer = q.get("answer", "")
            answer_type = q.get("answer_type", "")
            scale = q.get("scale", "")
            derivation = q.get("derivation", "")
            answer_from = q.get("answer_from", "")
            uid = q.get("uid", f"tatqa_{split}_{len(records)}")

            # Normalise answer to a readable string
            if isinstance(answer, list):
                answer_str = "; ".join(str(a) for a in answer)
            else:
                answer_str = str(answer) if answer is not None else ""

            annotation = classify_severity(question_text, answer_type, scale, answer)

            records.append(
                {
                    "id": uid,
                    "question": question_text,
                    "answer": answer_str,
                    "severity": annotation["severity"],
                    "scale": annotation["scale"],
                    "answer_type": annotation["answer_type"],
                    "answer_from": answer_from,
                    "derivation": derivation,
                    "domain": "finance",
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# CLI entry-point for quick sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load TAT-QA dataset")
    parser.add_argument("--split", default="test", choices=list(_SPLITS))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    df = load_tatqa(split=args.split, limit=args.limit)
    print(df.to_string(max_colwidth=80))
    print(f"\nShape: {df.shape}")
    print("\nSeverity distribution:")
    print(df["severity"].value_counts())
