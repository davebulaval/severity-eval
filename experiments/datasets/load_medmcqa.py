"""Load and annotate MedMCQA dataset with severity levels.

MedMCQA: ~194k 4-option MCQ questions from Indian medical entrance exams
(NEET-PG, AIIMS). Curated by domain experts, each question is tagged with
the medical specialty (`subject_name`) the practitioner is examined on.

Source: openlifescienceai/medmcqa on HuggingFace (parquet, public).
License: MIT.

Severity is assigned per question via the `subject_name` field, following
the same domain-expert-derived pattern as HEAD-QA: specialties carry
inherent differential financial-cost-of-error exposure based on the
clinical-decision impact of a wrong answer in deployment (drug dosing
errors > diagnostic delays > textbook anatomy mistakes). The mapping
mirrors HEAD-QA's category logic and is grounded in published diagnostic-
error cost data (Singh et al. 2014, Newman-Toker et al. 2019).

Cost vector: medical ($500, $5k, $50k, $500k).
"""

from __future__ import annotations

import pandas as pd

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


# Subject -> severity. Tiered by clinical-decision financial exposure.
SUBJECT_SEVERITY: dict[str, str] = {
    # Critical: direct iatrogenic harm risk (drug, anesthesia, surgical errors)
    "pharmacology": "critical",
    "anaesthesia": "critical",
    "surgery": "critical",
    "medicine": "critical",
    "pediatrics": "critical",
    "gynaecology & obstetrics": "critical",
    "obstetrics & gynaecology": "critical",
    # Major: clinical-decision impact (diagnosis, prescribing, imaging)
    "pathology": "major",
    "microbiology": "major",
    "orthopaedics": "major",
    "radiology": "major",
    "ent": "major",
    # Minor: clinical relevance but lower stakes per single decision
    "forensic medicine": "minor",
    "ophthalmology": "minor",
    "psychiatry": "minor",
    "preventive & social medicine": "minor",
    "social & preventive medicine": "minor",
    "dental": "minor",
    "skin": "minor",
    # Negligible: foundational science, error rarely propagates to harm
    "anatomy": "negligible",
    "biochemistry": "negligible",
    "physiology": "negligible",
    "unknown": "negligible",
}


def classify_severity(subject: str) -> str:
    """Assign severity based on MedMCQA subject_name."""
    return SUBJECT_SEVERITY.get(subject.strip().lower(), "minor")


def load_medmcqa(limit: int | None = None) -> pd.DataFrame:
    """Load MedMCQA train split and annotate with severity.

    Parameters
    ----------
    limit : int or None
        Max number of instances to load (after shuffling? No — preserves
        dataset order so the same `--limit N` always selects the same N
        instances across runs).

    Returns
    -------
    DataFrame with columns:
        id, question, answer, severity, category, options, domain
    """
    if load_dataset is None:
        raise ImportError("Install the datasets package: pip install datasets")

    ds = load_dataset("openlifescienceai/medmcqa", split="train")

    # cop is the 0-indexed correct option; opa/opb/opc/opd are the choices.
    letter_for = {0: "A", 1: "B", 2: "C", 3: "D"}
    records: list[dict] = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break

        qid = row.get("id") or f"medmcqa_{i:06d}"
        question: str = (row.get("question") or "").strip()
        if not question:
            # Empty stems would generate a prompt the model cannot answer
            # and inflate the per-dataset error rate spuriously.
            continue
        cop_raw = row.get("cop")
        try:
            cop = int(cop_raw) if cop_raw is not None else None
        except (TypeError, ValueError):
            cop = None
        if cop is None or cop not in (0, 1, 2, 3):
            # Skip malformed rows rather than poison the eval set.
            continue
        opts = {
            "A": row.get("opa") or "",
            "B": row.get("opb") or "",
            "C": row.get("opc") or "",
            "D": row.get("opd") or "",
        }
        answer_letter = letter_for[cop]
        answer_text = opts[answer_letter]
        if not answer_text:
            # The correct option has no text; the eval cannot compute a
            # meaningful score against an empty reference.
            continue
        subject = (row.get("subject_name") or "Unknown").strip()

        records.append(
            {
                "id": f"medmcqa_{i:06d}_{qid}",
                "question": question,
                "answer": answer_letter,
                "answer_text": answer_text,
                "severity": classify_severity(subject),
                "category": subject,
                "options": opts,
                "domain": "medical",
            }
        )

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = load_medmcqa(limit=10)
    print(f"Loaded {len(df)} rows")
    print(df[["id", "category", "severity", "answer"]].to_string())
    print("\nSeverity distribution:")
    print(df["severity"].value_counts())
