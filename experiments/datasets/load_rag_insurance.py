"""Load and annotate the Quebec auto insurance RAG dataset with severity levels.

Dataset: 82 manually annotated questions from Beauchemin et al.
Domain: Insurance QA (French) — auto insurance regulations in Quebec.
Source: dataset/all_manual_evaluations.jsonl (project-relative)

Severity is assigned based on the legal/financial impact of the topic:
  - critical: liability limits, mandatory coverage amounts, penalties
  - major:    claims process, coverage conditions, exclusions
  - minor:    general coverage descriptions, optional endorsements
  - negligible: definitions, administrative procedures, terminology

Cost vector: Insurance domain ($200, $2k, $20k, $200k) — similar to legal.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Relative to project root (severity-eval/)
RAG_DATASET_PATH = Path(__file__).resolve().parents[2] / "dataset" / "all_manual_evaluations.jsonl"

# Keywords for severity classification based on insurance topic
CRITICAL_KEYWORDS = [
    "responsabilite civile",
    "responsabilité civile",
    "minimum",
    "minimale",
    "obligatoire",
    "50 000",
    "1 000 000",
    "amende",
    "penalite",
    "pénalité",
    "suspension",
    "permis",
    "indemnite",
    "indemnité",
    "indemnisation",
    "deces",
    "décès",
    "mort",
    "blessure corporelle",
    "dommage corporel",
    "blessé",
    "blessure",
    "responsable",
]

MAJOR_KEYWORDS = [
    "reclamation",
    "réclamation",
    "sinistre",
    "exclusion",
    "restriction",
    "franchise",
    "collision",
    "renversement",
    "volé",
    "vol ",
    "vol?",
    "vol.",  # avoid matching "volumineux"
    "incendie",
    "resiliation",
    "résiliation",
    "subrogation",
    "couverture",
    "garantie",
    "chapitre b",
    "perte totale",
    "dommage",
]

MINOR_KEYWORDS = [
    "avenant",
    "endorsement",
    "deplacement",
    "déplacement",
    "remplacement",
    "location",
    "optionnel",
    "facultatif",
    "renouvellement",
    "prime",
]

NEGLIGIBLE_KEYWORDS = [
    "definition",
    "définition",
    "signifie",
    "veut dire",
    "que signifie",
    "formulaire",
    "procedure",
    "procédure",
    "quel organisme",
    "rôle de l'organisme",
    "couleur",
    "coûtent-elles plus cher",
]


def _classify_insurance_severity(question: str) -> str:
    """Classify severity based on the insurance topic of the question."""
    q_lower = question.lower()

    for keyword in CRITICAL_KEYWORDS:
        if keyword in q_lower:
            return "critical"

    for keyword in MAJOR_KEYWORDS:
        if keyword in q_lower:
            return "major"

    for keyword in MINOR_KEYWORDS:
        if keyword in q_lower:
            return "minor"

    for keyword in NEGLIGIBLE_KEYWORDS:
        if keyword in q_lower:
            return "negligible"

    # Default: minor (most insurance questions have moderate impact)
    return "minor"


def load_rag_insurance(
    limit: int | None = None,
    path: str | Path | None = None,
) -> pd.DataFrame:
    """Load the Quebec auto insurance RAG dataset with severity annotations.

    Parameters
    ----------
    limit : int or None
        Max number of instances to load.
    path : str or Path or None
        Override path to the JSONL file.

    Returns
    -------
    DataFrame with columns:
        id, question, answer, severity, llm_answer, score,
        human_judgment, domain
    """
    data_path = Path(path) if path else RAG_DATASET_PATH

    if not data_path.exists():
        raise FileNotFoundError(
            f"RAG dataset not found at {data_path}. Place the file in dataset/ at the project root."
        )

    records = []
    with open(data_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break

            row = json.loads(line)
            question = row.get("question", "")
            answer = row.get("output.model_output.llm_answer", "")
            score = row.get("score", None)
            human_judgment = row.get("answer", "")

            severity = _classify_insurance_severity(question)

            records.append(
                {
                    "id": f"rag_insurance_{i:04d}",
                    "question": question,
                    "answer": answer,
                    "severity": severity,
                    "score": score,
                    "human_judgment": human_judgment,
                    "domain": "insurance",
                }
            )

    return pd.DataFrame(records)
