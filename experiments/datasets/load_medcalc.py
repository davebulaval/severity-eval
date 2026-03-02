"""Load and annotate MedCalc-Bench dataset with severity levels.

MedCalc-Bench: 11,643 medical calculation instances.
Severity is assigned based on the calculator category.
"""

from __future__ import annotations

import pandas as pd

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

# Severity mapping based on calculator type / medical domain.
# Within each level, more specific multi-word keywords should appear before
# shorter ones so substring matching works correctly.
SEVERITY_RULES = {
    "critical": [
        "dosage",
        "dose",
        "sepsis",
        "triage",
        "sofa",
        "apache",
        "glasgow",
        "meld",
        "child-pugh",
        "wells",
        "perc ",
        "anion gap",
        "corrected calcium",
        "creatinine clearance",
    ],
    "major": [
        "surgical",
        "risk score",
        "mortality",
        "prognosis",
        "chadsvasc",
        "cha2ds2",
        "has-bled",
        "curb-65",
        "psi score",
        "fine score",
        "psi/fine",
        "alvarado",
        "bishop",
        "apgar",
    ],
    "minor": [
        "framingham risk score",
        "framingham",
        "ascvd",
        "reynolds",
        "lipid",
        "ldl",
        "cholesterol",
        "a1c",
        "glucose",
        "potassium",
        "sodium",
        "electrolyte",
    ],
    "negligible": [
        "bmi",
        "bsa",
        "body surface",
        "body mass",
        "conversion",
        "ideal body weight",
        "ibw",
        "corrected weight",
        "lean body",
    ],
}


def classify_severity(calculator_name: str, question: str) -> str:
    """Assign severity based on the medical calculator type.

    Keywords are matched longest-first across all levels so that specific
    multi-word entries (e.g. "framingham risk score") take priority over
    shorter ones (e.g. "risk score") regardless of severity level ordering.
    """
    text = (calculator_name + " " + question).lower()

    # Build a flat list of (keyword, level) sorted by keyword length descending
    # so more specific keywords always win over broader substrings.  When two
    # keywords have the same length, the most severe level wins as a
    # tiebreaker: critical > major > minor > negligible.
    _level_priority = {"critical": 3, "major": 2, "minor": 1, "negligible": 0}
    candidates = [(kw, level) for level, keywords in SEVERITY_RULES.items() for kw in keywords]
    candidates.sort(key=lambda x: (len(x[0]), _level_priority[x[1]]), reverse=True)

    for keyword, level in candidates:
        if keyword in text:
            return level

    return "major"  # default for unclassified medical calculations


def load_medcalc(limit: int | None = None) -> pd.DataFrame:
    """Load MedCalc-Bench and annotate with severity.

    Parameters
    ----------
    limit : int or None
        Max number of instances to load.

    Returns
    -------
    DataFrame with columns: question, answer, severity, domain
    """
    if load_dataset is None:
        raise ImportError("Install datasets: pip install datasets")

    ds = load_dataset("ncbi/MedCalc-Bench", split="test")

    records = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break

        calculator = row.get("Calculator Name", row.get("calculator_name", ""))
        question = row.get("Patient Note", row.get("question", ""))
        answer = str(row.get("Ground Truth Answer", row.get("answer", "")))
        severity = classify_severity(calculator, question)

        records.append(
            {
                "id": f"medcalc_{i}",
                "question": question,
                "answer": answer,
                "calculator": calculator,
                "severity": severity,
                "domain": "medical",
            }
        )

    return pd.DataFrame(records)
