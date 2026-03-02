"""Load and annotate HEAD-QA dataset with severity levels.

HEAD-QA: 6,765+ healthcare exam MCQ questions in English.
Source: alesi12/head_qa_v2 on HuggingFace (parquet-based, replaces the
        deprecated dvilares/head_qa which relies on a loading script no
        longer supported by the datasets library).
License: MIT.

Severity is assigned based on the exam category field.
Cost vector: medical ($500, $5k, $50k, $500k).
"""

from __future__ import annotations

import pandas as pd

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

# Category-to-severity mapping
CATEGORY_SEVERITY: dict[str, str] = {
    "pharmacology": "critical",
    "medicine": "major",
    "nursing": "major",
    "psychology": "minor",
    "biology": "negligible",
    "chemistry": "negligible",
}


def classify_severity(category: str) -> str:
    """Assign severity based on HEAD-QA category."""
    return CATEGORY_SEVERITY.get(category.lower(), "minor")


def load_headqa(limit: int | None = None) -> pd.DataFrame:
    """Load HEAD-QA (English) and annotate with severity.

    Uses alesi12/head_qa_v2 which is stored as standard parquet files,
    unlike the original dvilares/head_qa which requires a custom loading
    script that is no longer supported.

    Parameters
    ----------
    limit : int or None
        Max number of instances to load.

    Returns
    -------
    DataFrame with columns:
        id, question, answer, severity, category, domain
    """
    if load_dataset is None:
        raise ImportError("Install the datasets package: pip install datasets")

    ds = load_dataset("alesi12/head_qa_v2", "en", split="train")

    records: list[dict] = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break

        qid: int = row.get("qid", i)
        question: str = row.get("qtext", "")
        category: str = row.get("category", "")
        ra: int = row.get("ra", -1)  # correct answer ID (1-indexed)
        answers: list[dict] = row.get("answers", [])

        # Resolve correct answer text from the answers list
        # aid may be int or str depending on parquet serialisation
        answer_text = ""
        for ans in answers:
            if str(ans.get("aid", "")) == str(ra):
                answer_text = ans.get("atext", "")
                break

        # Build options dict {1: text, 2: text, ...}
        options = {ans["aid"]: ans["atext"] for ans in answers}

        records.append(
            {
                "id": f"headqa_{qid}",
                "question": question,
                "answer": answer_text,
                "severity": classify_severity(category),
                "category": category,
                "options": options,
                "domain": "medical",
            }
        )

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = load_headqa(limit=5)
    print(f"Loaded {len(df)} rows")
    print(df[["id", "category", "severity", "answer"]].to_string())
    print("\nSeverity distribution:")
    print(df["severity"].value_counts())
