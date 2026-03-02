"""Load and annotate the JUDGEBERT text simplification dataset with severity levels.

Dataset: 1,484 annotated insurance text simplifications from Beauchemin et al.
Domain: Legal text simplification (French) — Quebec auto insurance contracts.
Source: dataset/insurance_text_simplifications_annotated.jsonl (project-relative)

Severity is assigned based on the legal category of the clause (from
the annotation's accepted option) and the evaluation score:
  - critical: high-impact clauses (exclusions, consequences) — base level;
              upgraded from major if evaluation score ≤ 4
  - major:    high-impact clauses (indemnities, obligations, subrogation,
              termination, recourse) — base level;
              upgraded from minor if evaluation score ≤ 4
  - minor:    medium-impact clauses (conditions, damages, fees, etc.)
              — base level; upgraded from negligible if score ≤ 4
  - negligible: definitions, descriptions — base level (not upgraded)

Cost vector: Legal/Insurance domain ($200, $2k, $20k, $200k).

The evaluation score (1-10) measures how well the simplification
preserves legal meaning. Lower scores = worse preservation = higher
severity if the simplified text were used in place of the original.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Relative to project root (severity-eval/)
JUDGEBERT_PATH = Path(__file__).resolve().parents[2] / "dataset" / "insurance_text_simplifications_annotated.jsonl"

# Legal clause categories and their base severity
# Based on the 18 option categories in the annotation schema
HIGH_IMPACT_CATEGORIES = {
    "3": "critical",  # Exclusion(s) ou restriction(s)
    "5": "major",  # Indemnités
    "9": "major",  # Obligation(s) de l'assuré
    "10": "critical",  # Conséquence(s) du non-respect des obligations
    "14": "major",  # Subrogation
    "16": "major",  # Fin du contrat, résiliation
    "17": "major",  # Recours
}

MEDIUM_IMPACT_CATEGORIES = {
    "2": "minor",  # Condition(s) d'application
    "4": "minor",  # Dommages
    "7": "minor",  # Frais
    "8": "minor",  # Prime
    "11": "minor",  # Obligation(s) de l'assureur
    "12": "minor",  # Droit(s) de l'assuré
    "13": "minor",  # Droit(s) de l'assureur
    "15": "minor",  # Prise d'effet, renouvellement
}

LOW_IMPACT_CATEGORIES = {
    "1": "negligible",  # Description de l'avenant/de la garantie
    "6": "negligible",  # Définition
    "18": "negligible",  # Autres
}

ALL_CATEGORIES = {**HIGH_IMPACT_CATEGORIES, **MEDIUM_IMPACT_CATEGORIES, **LOW_IMPACT_CATEGORIES}


def _classify_simplification_severity(
    accepted_categories: list[str],
    evaluation_score: int | float | None,
) -> str:
    """Classify severity based on clause category and evaluation quality.

    Parameters
    ----------
    accepted_categories : list[str]
        List of accepted category IDs from the annotation.
    evaluation_score : int or float or None
        Quality score (1-10) of the simplification.
        Lower = worse preservation of legal meaning.
    """
    # Determine base severity from the clause category
    base_severity = "minor"  # default
    if accepted_categories:
        cat_id = accepted_categories[0]
        base_severity = ALL_CATEGORIES.get(cat_id, "minor")

    # Adjust based on evaluation score
    if evaluation_score is not None:
        try:
            score = float(evaluation_score)
        except (ValueError, TypeError):
            return base_severity

        severity_levels = ["negligible", "minor", "major", "critical"]
        base_idx = severity_levels.index(base_severity)

        # Poor simplification (score 1-4): upgrade severity by 1
        if score <= 4:
            return severity_levels[min(base_idx + 1, 3)]
        # Good simplification (score 8-10): keep base severity
        elif score >= 8:
            return base_severity
        # Medium (5-7): keep base severity
        else:
            return base_severity

    return base_severity


def load_judgebert(
    limit: int | None = None,
    path: str | Path | None = None,
) -> pd.DataFrame:
    """Load the JUDGEBERT text simplification dataset with severity annotations.

    Parameters
    ----------
    limit : int or None
        Max number of instances to load.
    path : str or Path or None
        Override path to the JSONL file.

    Returns
    -------
    DataFrame with columns:
        id, question (original text), answer (simplified text), severity,
        category, evaluation, annotator, domain
    """
    data_path = Path(path) if path else JUDGEBERT_PATH

    if not data_path.exists():
        raise FileNotFoundError(
            f"JUDGEBERT dataset not found at {data_path}. Place the file in dataset/ at the project root."
        )

    records = []
    with open(data_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            original = row.get("text", row.get("original", ""))
            simplified = row.get("simplified_text", row.get("simplified", ""))
            evaluation = row.get("evaluation", None)
            accepted = row.get("accept", [])
            annotator = row.get("_annotator_id", "")

            severity = _classify_simplification_severity(accepted, evaluation)

            # Map category IDs to names
            options = row.get("options", [])
            category_names = []
            for cat_id in accepted:
                for opt in options:
                    if opt.get("id") == cat_id:
                        category_names.append(opt.get("text", cat_id))
                        break

            records.append(
                {
                    "id": f"judgebert_{i:04d}",
                    "question": original,
                    "answer": simplified,
                    "severity": severity,
                    "category": ", ".join(category_names) if category_names else "",
                    "evaluation": evaluation,
                    "annotator": annotator,
                    "domain": "legal_simplification",
                }
            )

    return pd.DataFrame(records)
