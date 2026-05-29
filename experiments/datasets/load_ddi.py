"""Load and annotate DDI Corpus (Drug-Drug Interactions) with severity.

DDI Corpus 2013 (Herrero-Zazo et al. 2013, "The DDI corpus: An annotated
corpus with pharmacological substances and drug-drug interactions").
Annotated by clinical pharmacology experts at UC3M; gold labels are
per-pair and grounded in DrugBank / pharmacology references.

Source: the official XML release, mirrored at
        https://github.com/isegura/DDICorpus (Isabel Segura-Bedmar).
        Place the unzipped corpus under ``dataset/ddi_corpus/`` so the
        loader finds ``**/*.xml`` recursively.
License: CC BY-NC-SA 4.0.

The corpus is distributed as XML files (SemEval-2013 Task 9 format) and
the official ``bigbio/ddi_corpus`` Hugging Face dataset is script-based,
which the current ``datasets`` library no longer loads. To keep the
loader self-contained and reproducible, we parse the XML directly.

Each annotated pair (drug1, drug2, sentence) carries one of five labels:
    - effect    : interaction with described clinical effect
    - mechanism : interaction with described pharmacokinetic mechanism
    - advice    : recommendation about co-administration
    - int       : interaction mentioned without further classification
    - false     : no interaction asserted (ddi="false" pairs in the XML)

Severity is the financial cost of misclassifying that pair, anchored on
published costs of preventable adverse drug events (Bates et al. 1997;
Classen et al. 1997; AHRQ 2020). Missing an "effect" pair in a clinical
decision-support context is the most expensive error (preventable ADE
hospitalization, US$~50k median; fatal ADE exposure US$M-scale via
malpractice). Missing "advice" or "mechanism" is intermediate
(suboptimal prescribing). "int" without specification is minor signal.
A "false" gold label means the pair carries no interaction, so the cost
of a model error is a spurious alert (low operational cost).

Cost vector: medical ($500, $5k, $50k, $500k).

Task framing: given a sentence and two drug mentions, the model is asked
to classify the relation as one of {none, mechanism, effect, advice, int}.
The gold label maps to severity per the table above; the model's
prediction is scored as exact match against the gold class letter.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


# Project-relative path to the unzipped DDI XML release.
DDI_XML_DIR = (
    Path(__file__).resolve().parents[2] / "dataset" / "ddi_corpus"
)


# Relation type -> severity tier.
RELATION_SEVERITY: dict[str, str] = {
    "effect": "critical",
    "advice": "major",
    "mechanism": "major",
    "int": "minor",
    "false": "negligible",
    # Tolerate the DDI- prefixes used by some derivatives.
    "ddi-effect": "critical",
    "ddi-advise": "major",
    "ddi-mechanism": "major",
    "ddi-int": "minor",
    "ddi-false": "negligible",
    # Sentinel used by the loader for "no interaction".
    "none": "negligible",
}

_RELATION_LETTER: dict[str, str] = {
    "none": "A",
    "mechanism": "B",
    "effect": "C",
    "advice": "D",
    "int": "E",
}

_OPTIONS = {
    "A": "No interaction",
    "B": "Mechanism (pharmacokinetic interaction described)",
    "C": "Effect (clinical effect of co-administration described)",
    "D": "Advice (recommendation about co-administration)",
    "E": "Int (interaction mentioned without further specification)",
}


def _normalise_relation(label: str) -> str:
    """Map dataset-specific relation strings to the 5-class space."""
    lab = (label or "").strip().lower()
    if lab.startswith("ddi-"):
        lab = lab[4:]
    if lab in ("advise", "advices"):
        lab = "advice"
    if lab in ("", "false", "none", "no", "n"):
        return "none"
    if lab in {"mechanism", "effect", "advice", "int"}:
        return lab
    return "none"


def classify_severity(relation: str) -> str:
    """Severity for a normalised relation label."""
    return RELATION_SEVERITY.get(_normalise_relation(relation), "minor")


def _resolve_xml_dir(path: str | Path | None) -> Path:
    base = Path(path) if path else DDI_XML_DIR
    if not base.exists():
        raise FileNotFoundError(
            f"DDI corpus not found at {base}. Download from "
            "https://github.com/isegura/DDICorpus (unzip into "
            "dataset/ddi_corpus/) and retry."
        )
    if not any(base.rglob("*.xml")):
        raise FileNotFoundError(
            f"No XML files under {base}; the directory exists but is empty. "
            "Re-extract the DDI corpus zip there."
        )
    return base


def _iter_pairs_from_xml(xml_dir: Path) -> list[dict]:
    """Walk all XML files under ``xml_dir`` and emit one record per
    annotated pair.

    The DDI Corpus XML schema (SemEval-2013 Task 9):

        <document id="...">
          <sentence id="..." text="...">
            <entity id="..." type="drug" charOffset="0-12" text="..."/>
            ...
            <pair id="..." e1="..." e2="..." ddi="true" type="effect"/>
            ...
          </sentence>
        </document>

    The ``type`` attribute is only present when ``ddi="true"``; pairs
    flagged ``ddi="false"`` carry no relation type and are emitted with
    the ``none`` sentinel.
    """
    records: list[dict] = []
    for xml_path in sorted(xml_dir.rglob("*.xml")):
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            continue
        root = tree.getroot()
        doc_id = root.get("id") or xml_path.stem
        sentences = root.findall(".//sentence") or [root]
        for sent in sentences:
            sent_text = sent.get("text") or ""
            entities: dict[str, str] = {}
            for ent in sent.findall("entity"):
                eid = ent.get("id")
                if not eid:
                    continue
                entities[eid] = ent.get("text") or ""
            for pair in sent.findall("pair"):
                e1 = pair.get("e1")
                e2 = pair.get("e2")
                drug1 = entities.get(e1, "")
                drug2 = entities.get(e2, "")
                if not drug1 or not drug2:
                    continue
                ddi_flag = (pair.get("ddi") or "false").strip().lower() == "true"
                rel_type = pair.get("type") if ddi_flag else "false"
                relation = _normalise_relation(rel_type or "false")
                records.append(
                    {
                        "doc_id": doc_id,
                        "sentence_id": sent.get("id") or "",
                        "pair_id": pair.get("id") or "",
                        "drug1": drug1,
                        "drug2": drug2,
                        "sentence": sent_text,
                        "relation": relation,
                    }
                )
    return records


def load_ddi(
    limit: int | None = None,
    xml_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load DDI Corpus 2013 from local XML and annotate per-pair severity.

    Parameters
    ----------
    limit : int or None
        Max number of pairs to load.
    xml_dir : path or None
        Override directory containing the unzipped XML release.

    Returns
    -------
    DataFrame with columns:
        id, question, answer, severity, category, options, evidence, domain
    """
    base = _resolve_xml_dir(xml_dir)
    pairs = _iter_pairs_from_xml(base)

    records: list[dict] = []
    for i, p in enumerate(pairs):
        if limit is not None and i >= limit:
            break
        rel = _normalise_relation(p["relation"])
        answer_letter = _RELATION_LETTER[rel]
        question = (
            f"In the sentence below, classify the interaction between "
            f"\"{p['drug1']}\" and \"{p['drug2']}\"."
        )
        # Build a stable id from the corpus' own pair id when present.
        rid = p["pair_id"] or f"{p['doc_id']}_{p['sentence_id']}_{i:06d}"
        records.append(
            {
                "id": f"ddi_{i:06d}_{rid}",
                "question": question,
                "answer": answer_letter,
                "answer_text": _OPTIONS[answer_letter],
                "severity": classify_severity(rel),
                "category": rel,
                "options": dict(_OPTIONS),
                "evidence": p["sentence"],
                "domain": "medical",
            }
        )

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = load_ddi(limit=10)
    print(f"Loaded {len(df)} rows")
    print(df[["id", "category", "severity", "answer"]].to_string())
    print("\nSeverity distribution:")
    print(df["severity"].value_counts())
