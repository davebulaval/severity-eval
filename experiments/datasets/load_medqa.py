"""Load and annotate MedQA USMLE dataset with severity levels.

MedQA USMLE: ~11,400 USMLE-style MCQ questions (train + test splits).
Source: GBaker/MedQA-USMLE-4-options on HuggingFace.
License: MIT.

Severity is assigned based on clinical domain keywords in the question text.
Cost vector: medical ($500, $5k, $50k, $500k).
"""

from __future__ import annotations

import pandas as pd

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

# Keyword-based severity rules, applied in priority order (critical first)
SEVERITY_RULES: dict[str, list[str]] = {
    "critical": [
        # Pharmacology / dosage
        "dose",
        "drug",
        "medication",
        "prescribe",
        "toxicity",
        "overdose",
        "contraindication",
        # Emergency / critical care
        "emergency",
        "icu",
        "resuscitate",
        "shock",
        "cardiac arrest",
    ],
    "major": [
        # Diagnosis
        "diagnosis",
        "most likely",
        "differential",
        # Surgical
        "surgery",
        "surgical",
        "operative",
        "resection",
    ],
    "minor": [
        # Pathophysiology / histology
        "mechanism",
        "pathology",
        "histology",
        # Lab interpretation
        "lab",
        "blood test",
        "cbc",
        "serum",
    ],
    "negligible": [
        # Anatomy
        "anatomy",
        "nerve",
        "artery",
        "muscle",
        # Basic science
        "embryology",
        "genetics",
    ],
}


def classify_severity(question: str) -> str:
    """Assign severity based on clinical keyword matches in the question."""
    text = question.lower()
    for level in ("critical", "major", "minor", "negligible"):
        for keyword in SEVERITY_RULES[level]:
            if keyword in text:
                return level
    return "major"  # default for unclassified clinical questions


def load_medqa(limit: int | None = None) -> pd.DataFrame:
    """Load MedQA USMLE and annotate with severity.

    Combines train and test splits from GBaker/MedQA-USMLE-4-options.

    Parameters
    ----------
    limit : int or None
        Max number of instances to load (across combined splits).

    Returns
    -------
    DataFrame with columns:
        id, question, answer, severity, options, domain
    """
    if load_dataset is None:
        raise ImportError("Install the datasets package: pip install datasets")

    splits = ["train", "test"]
    records: list[dict] = []
    global_idx = 0

    for split in splits:
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split=split)

        for row in ds:
            if limit is not None and global_idx >= limit:
                break

            question: str = row.get("question", "")
            answer: str = row.get("answer", "")
            options: dict = row.get("options", {})
            meta_info: str = row.get("meta_info", "")

            records.append(
                {
                    "id": f"medqa_{global_idx}",
                    "question": question,
                    "answer": answer,
                    "severity": classify_severity(question),
                    "options": options,
                    "meta_info": meta_info,
                    "domain": "medical",
                }
            )
            global_idx += 1

        if limit is not None and global_idx >= limit:
            break

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = load_medqa(limit=5)
    print(f"Loaded {len(df)} rows")
    print(df[["id", "severity", "meta_info", "answer"]].to_string())
    print("\nSeverity distribution:")
    print(df["severity"].value_counts())
