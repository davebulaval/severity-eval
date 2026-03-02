"""Load and annotate CUAD dataset with severity levels.

CUAD: 13,000+ contract clauses from legal documents.
Source: https://github.com/TheAtticusProject/cuad (SQuAD format)
License: CC BY 4.0

Note: the theatticusproject/cuad-qa HuggingFace repo uses a deprecated
loading script that is no longer supported by the datasets library.
This loader downloads the zip archive directly from the GitHub repo.

Severity is assigned based on the clause type.
"""

from __future__ import annotations

import io
import json
import urllib.request
import zipfile

import pandas as pd

_DATA_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"

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


def _download_cuad() -> dict:
    """Download and parse the CUAD SQuAD-format JSON from GitHub."""
    try:
        with urllib.request.urlopen(_DATA_URL, timeout=120) as resp:
            data = resp.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download CUAD from {_DATA_URL}: {exc}") from exc

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        # The zip contains test.json and train.json in SQuAD format
        for name in zf.namelist():
            if name.endswith("test.json"):
                with zf.open(name) as f:
                    return json.loads(f.read().decode("utf-8"))

    raise RuntimeError("test.json not found in CUAD data.zip")


def load_cuad(limit: int | None = None) -> pd.DataFrame:
    """Load CUAD and annotate with severity.

    Downloads data directly from the GitHub repo (SQuAD format) since
    the HuggingFace dataset script is no longer supported.

    Parameters
    ----------
    limit : int or None
        Max number of instances to load.

    Returns
    -------
    DataFrame with columns: id, question, answer, evidence, clause_type,
        severity, domain
    """
    raw = _download_cuad()

    records = []
    for article in raw.get("data", []):
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "")
            for qa in paragraph.get("qas", []):
                if limit and len(records) >= limit:
                    break

                question = qa.get("question", "")
                answers = qa.get("answers", [])
                answer_text = answers[0].get("text", "") if answers else ""
                qa_id = qa.get("id", f"cuad_{len(records)}")

                # Match against known clause types
                matched_type = None
                for known_type in sorted(CLAUSE_SEVERITY, key=len, reverse=True):
                    if known_type.lower() in question.lower():
                        matched_type = known_type
                        break

                severity = classify_severity(matched_type) if matched_type else "minor"

                records.append(
                    {
                        "id": qa_id,
                        "question": question,
                        "answer": answer_text,
                        "evidence": context,
                        "clause_type": matched_type or "unknown",
                        "severity": severity,
                        "domain": "legal",
                    }
                )

            if limit and len(records) >= limit:
                break
        if limit and len(records) >= limit:
            break

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = load_cuad(limit=5)
    print(df[["id", "question", "answer", "clause_type", "severity"]].to_string(max_colwidth=60))
    print(f"\nEvidence present: {'evidence' in df.columns}")
    print(f"Shape: {df.shape}")
