"""Load and annotate CUAD dataset with severity levels.

CUAD: 13,000+ contract clauses from legal documents.
Severity is assigned based on the clause type.
"""

from __future__ import annotations

import pandas as pd

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

# Severity mapping based on clause type.
# Keys are sorted longest-first within each group so that substring matching
# (e.g. "Cap On Liability" before "Liability") resolves correctly.
# The dict itself is iterated in insertion order (Python 3.7+), so the order
# here matters for the matching loop in load_cuad().
CLAUSE_SEVERITY = {
    # Critical: high financial/legal exposure
    "Termination For Convenience": "critical",
    "Termination For Cause": "critical",
    "Limitation Of Liability": "critical",
    "Indemnification": "critical",
    "Insurance": "critical",
    "Liability": "critical",
    # Major: significant business impact
    "Notice Period To Terminate Renewal": "major",
    "Ip Ownership Assignment": "major",
    "Revenue/Profit Sharing": "major",
    "Most Favored Nation": "major",
    "Competitive Restriction Exception": "major",
    "Irrevocable Or Perpetual License": "major",
    "Change Of Control": "major",
    "Minimum Commitment": "major",
    "Price Restrictions": "major",
    "Uncapped Liability": "critical",
    "Cap On Liability": "major",
    "Anti-Assignment": "major",
    "Audit Rights": "major",
    "Non-Compete": "major",
    "License Grant": "major",
    "Exclusivity": "major",
    "Warranty": "major",
    # Minor: operational/administrative
    "Post-Termination Services": "minor",
    "Third Party Beneficiary": "minor",
    "Covenant Not To Sue": "major",
    "Dispute Resolution": "major",
    "Governing Law": "minor",
    "Renewal Term": "minor",
    "Effective Date": "minor",
    "Expiration Date": "minor",
    # Negligible: boilerplate/low risk
    "Affiliate License-Licensee": "negligible",
    "Affiliate License-Licensor": "negligible",
    "Source Code Escrow": "negligible",
    "Volume Restriction": "negligible",
    "Joint Ip Ownership": "major",
    "Rofr/Rofo/Rofn": "minor",
    "Agreement Date": "negligible",
    "Document Name": "negligible",
    "Parties": "negligible",
}


def classify_severity(clause_type: str) -> str:
    """Assign severity based on clause type."""
    return CLAUSE_SEVERITY.get(clause_type, "minor")


def load_cuad(limit: int | None = None) -> pd.DataFrame:
    """Load CUAD and annotate with severity.

    Parameters
    ----------
    limit : int or None
        Max number of instances to load.

    Returns
    -------
    DataFrame with columns: question, answer, severity, domain, clause_type
    """
    if load_dataset is None:
        raise ImportError("Install datasets: pip install datasets")

    ds = load_dataset("theatticusproject/cuad-qa", split="test")

    records = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break

        question = row.get("question", "")
        answers = row.get("answers", {})
        answer_text = answers.get("text", [""])[0] if isinstance(answers, dict) else ""

        # Match against known clause types. Iterate longest keys first so that
        # specific multi-word clauses (e.g. "Cap On Liability") are matched
        # before shorter substrings they contain (e.g. "Liability").
        matched_type = None
        for known_type in sorted(CLAUSE_SEVERITY, key=len, reverse=True):
            if known_type.lower() in question.lower():
                matched_type = known_type
                break

        severity = classify_severity(matched_type) if matched_type else "minor"

        records.append(
            {
                "id": f"cuad_{i}",
                "question": question,
                "answer": answer_text,
                "clause_type": matched_type or "unknown",
                "severity": severity,
                "domain": "legal",
            }
        )

    return pd.DataFrame(records)
