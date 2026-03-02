"""Load and annotate MAUD dataset with severity levels.

MAUD: Merger Agreement Understanding Dataset
~39,000 annotations across 152 merger agreements.
Source: theatticusproject/maud on HuggingFace
License: CC BY 4.0

Severity is assigned based on the deal point category and question type:
  - critical: MAE, termination fees, fiduciary duties, reverse termination
  - major:    matching rights, go-shop, specific performance, non-solicitation
  - minor:    representations, warranties, closing conditions, interim covenants
  - negligible: definitions, choice of law, general information

Cost vector: Legal domain ($200, $2k, $20k, $200k).
"""

from __future__ import annotations

import pandas as pd

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


# ──────────────────────────────────────────────────────────────────────────────
# Severity by MAUD category (top-level grouping)
# ──────────────────────────────────────────────────────────────────────────────
CATEGORY_SEVERITY: dict[str, str] = {
    # Critical: deal-breaker provisions with direct financial exposure
    "Material Adverse Effect": "critical",
    "Remedies": "critical",
    # Major: significant deal protection and negotiation levers
    "Deal Protection and Related Provisions": "major",
    # Minor: operational / closing mechanics
    "Conditions to Closing": "minor",
    "Operating and Efforts Covenant": "minor",
    # Negligible: definitional / boilerplate
    "General Information": "negligible",
    "Knowledge": "negligible",
}

# Fine-grained overrides keyed on question text (substring match, case-insensitive).
# Evaluated before category-level mapping; longest keys take priority.
QUESTION_SEVERITY_OVERRIDES: dict[str, str] = {
    # Critical — MAE-related
    "mae definition": "critical",
    "mae forward looking": "critical",
    "mae applies": "critical",
    "fls (mae)": "critical",
    "ability to consummate": "critical",
    # Critical — termination / fees
    "termination fee": "critical",
    "reverse termination": "critical",
    "tail period": "critical",
    # Critical — fiduciary / board
    "fiduciary": "critical",
    "change in recommendation": "critical",
    "cor permitted": "critical",
    "cor standard": "critical",
    # Major — go-shop / no-shop
    "no-shop": "major",
    "go-shop": "major",
    "no shop": "major",
    "acquisition proposal": "major",
    "ftr triggers": "major",
    "matching rights": "major",
    "initial matching rights": "major",
    "additional matching rights": "major",
    "intervening event": "major",
    "superior offer": "major",
    # Major — specific performance / remedies
    "specific performance": "major",
    # Minor — closing conditions / covenants
    "closing condition": "minor",
    "compliance with": "minor",
    "ordinary course": "minor",
    "negative interim covenant": "minor",
    "interim covenant": "minor",
    "antitrust efforts": "minor",
    "accuracy of": "minor",
    "materiality/mae scrape": "minor",
    "materiality scrape": "minor",
    # Negligible — definitional
    "knowledge definition": "negligible",
    "constructive knowledge": "negligible",
    "type of consideration": "negligible",
    "choice of law": "negligible",
    "governing law": "negligible",
    "definition contains": "negligible",
    "definitions": "negligible",
    "pandemic": "minor",
    "covid": "minor",
    "war, terrorism": "minor",
    "force majeure": "minor",
    "general economic": "negligible",
    "general political": "negligible",
    "changes in gaap": "negligible",
    "changes in market price": "negligible",
    "change in law": "negligible",
    "failure to meet projections": "negligible",
    "absence of litigation": "minor",
    "announcement, pendency": "negligible",
}


def _classify_severity(category: str, question: str) -> str:
    """Return severity for a MAUD deal point.

    Applies question-level overrides first (longest match wins), then falls
    back to category-level mapping, then defaults to 'minor'.
    """
    question_lower = question.lower()
    # Sort by key length descending so specific phrases win over substrings.
    for keyword in sorted(QUESTION_SEVERITY_OVERRIDES, key=len, reverse=True):
        if keyword in question_lower:
            return QUESTION_SEVERITY_OVERRIDES[keyword]

    return CATEGORY_SEVERITY.get(category, "minor")


def _build_answer_options(ds) -> dict[str, dict[str, str]]:
    """Pre-compute valid answer choices per question from the full dataset.

    Returns a dict mapping question text to {letter: answer_text}.
    """
    from collections import defaultdict

    answers_per_q: dict[str, set[str]] = defaultdict(set)
    for split_name in ds:
        for row in ds[split_name]:
            q = row.get("question", "")
            a = row.get("answer", "")
            if q and a:
                answers_per_q[q].add(a)

    options_map = {}
    for q, answers in answers_per_q.items():
        sorted_answers = sorted(answers)
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        options_map[q] = {letters[i]: a for i, a in enumerate(sorted_answers) if i < len(letters)}
    return options_map


def load_maud(limit: int | None = None) -> pd.DataFrame:
    """Load MAUD and annotate with severity.

    Combines train, validation and test splits into a single DataFrame.

    Parameters
    ----------
    limit : int or None
        Max number of instances to load (across all splits combined).

    Returns
    -------
    DataFrame with columns:
        id, question, answer, evidence, severity, category, options,
        contract_name, domain
    """
    if load_dataset is None:
        raise ImportError("Install datasets: pip install datasets")

    ds = load_dataset("theatticusproject/maud")
    answer_options = _build_answer_options(ds)

    records: list[dict] = []
    global_idx = 0

    for split_name in ("train", "validation", "test"):
        if split_name not in ds:
            continue
        for row in ds[split_name]:
            if limit is not None and global_idx >= limit:
                break

            question = row.get("question", "")
            subquestion = row.get("subquestion", "")
            category = row.get("category", "")
            answer = row.get("answer", "")
            contract_name = row.get("contract_name", "")
            text = row.get("text", "")
            row_id = row.get("id", str(global_idx))

            # Combine question + subquestion for richer context when classifying.
            full_question = question
            if subquestion and subquestion.strip() not in ("<NONE>", ""):
                full_question = f"{question} | {subquestion}"

            severity = _classify_severity(category, full_question)

            # Get MCQ options for this question type
            options = answer_options.get(question, {})

            records.append(
                {
                    "id": f"maud_{row_id}",
                    "question": full_question,
                    "answer": answer,
                    "evidence": text,
                    "options": options,
                    "severity": severity,
                    "category": category,
                    "contract_name": contract_name,
                    "domain": "legal",
                }
            )
            global_idx += 1

        if limit is not None and global_idx >= limit:
            break

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = load_maud(limit=5)
    print(df.to_string())
    print()
    print("Columns:", list(df.columns))
    print("Severity counts:", df["severity"].value_counts().to_dict())
