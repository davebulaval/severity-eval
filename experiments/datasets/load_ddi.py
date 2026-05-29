"""Load and annotate DDI Corpus (Drug-Drug Interactions) with severity.

DDI Corpus 2013 (Herrero-Zazo et al. 2013, "The DDI corpus: An annotated
corpus with pharmacological substances and drug-drug interactions").
Annotated by clinical pharmacology experts at UC3M; gold labels are
per-pair and grounded in DrugBank / pharmacology references.

Source: bigbio/ddi_corpus on HuggingFace (kb config).
License: CC BY-NC-SA 4.0.

Each annotated pair (drug1, drug2, sentence) carries one of five relation
labels, expert-graded by clinical pharmacologists:
    - effect    : a drug interaction with described clinical effect
    - mechanism : interaction with described pharmacokinetic mechanism
    - advice    : a recommendation about co-administration
    - int       : interaction mentioned without further classification
    - false     : no interaction asserted

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

import itertools

import pandas as pd

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


# Relation type -> severity tier.
RELATION_SEVERITY: dict[str, str] = {
    "effect": "critical",
    "advice": "major",
    "mechanism": "major",
    "int": "minor",
    "false": "negligible",
    # bigbio_kb sometimes returns "DDI-effect", "DDI-advise" etc.
    "ddi-effect": "critical",
    "ddi-advise": "major",
    "ddi-mechanism": "major",
    "ddi-int": "minor",
    "ddi-false": "negligible",
    # Sentinel for negative (non-interacting) pairs
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
    if lab in ("", "false", "none"):
        return "none"
    if lab in {"mechanism", "effect", "advice", "int"}:
        return lab
    return "none"


def classify_severity(relation: str) -> str:
    """Severity for a normalised relation label."""
    return RELATION_SEVERITY.get(_normalise_relation(relation), "minor")


def _iter_pairs_from_bigbio(ds) -> list[dict]:
    """Flatten bigbio_kb-formatted documents into (drug1, drug2, sentence,
    relation) records.

    bigbio_kb structure per doc:
        passages -> [{text, offsets, ...}]
        entities -> [{id, type, text, offsets}]
        relations -> [{type, arg1_id, arg2_id, ...}]

    Negative (non-interacting) pairs are NOT typically encoded as
    relations; we generate them by enumerating drug-drug entity pairs that
    appear in the same sentence and do not appear among the positive
    relations.
    """
    records: list[dict] = []
    for doc in ds:
        passages = doc.get("passages") or []
        entities = doc.get("entities") or []
        relations = doc.get("relations") or []

        # Index entities by id; keep only drug-like entity types.
        ent_by_id = {}
        for e in entities:
            eid = e.get("id")
            if not eid:
                continue
            # bigbio offsets can be nested lists [[start, end]]
            offsets = e.get("offsets") or []
            if offsets and isinstance(offsets[0], list):
                offsets = offsets[0]
            ent_by_id[eid] = {
                "id": eid,
                "type": e.get("type", ""),
                "text": (e.get("text") or [""])[0]
                if isinstance(e.get("text"), list)
                else (e.get("text") or ""),
                "offsets": offsets,
            }

        # Map each entity to its containing passage text (use earliest
        # passage covering the offset).
        passage_texts = [p.get("text") or "" for p in passages]
        passage_offsets = [p.get("offsets") or [] for p in passages]

        def _which_passage(off: list[int]) -> tuple[int, str] | None:
            if not off:
                return None
            start = off[0]
            for i, po in enumerate(passage_offsets):
                if not po:
                    continue
                rng = po[0] if isinstance(po[0], list) else po
                if len(rng) == 2 and rng[0] <= start < rng[1]:
                    text = passage_texts[i]
                    text = text[0] if isinstance(text, list) else text
                    return i, text
            return None

        # Positive pairs from relations
        pos_keys: set[tuple[str, str]] = set()
        for r in relations:
            rtype = r.get("type", "")
            a1 = r.get("arg1_id") or r.get("head", {}).get("ref_id")
            a2 = r.get("arg2_id") or r.get("tail", {}).get("ref_id")
            if not a1 or not a2 or a1 not in ent_by_id or a2 not in ent_by_id:
                continue
            pos_keys.add(tuple(sorted([a1, a2])))
            p = _which_passage(ent_by_id[a1]["offsets"])
            sent = p[1] if p else (passage_texts[0] if passage_texts else "")
            records.append(
                {
                    "drug1": ent_by_id[a1]["text"],
                    "drug2": ent_by_id[a2]["text"],
                    "sentence": sent if isinstance(sent, str) else "",
                    "relation": _normalise_relation(rtype),
                    "doc_id": doc.get("document_id") or doc.get("id", ""),
                }
            )

        # Negative pairs: drug-drug entity pairs in the same passage that
        # are not in pos_keys. Limit to keep dataset balanced.
        drug_ents = [
            e for e in ent_by_id.values() if "drug" in e["type"].lower() or e["type"] in ("DRUG", "GROUP", "BRAND")
        ]
        for e1, e2 in itertools.combinations(drug_ents, 2):
            key = tuple(sorted([e1["id"], e2["id"]]))
            if key in pos_keys:
                continue
            p1 = _which_passage(e1["offsets"])
            p2 = _which_passage(e2["offsets"])
            if not p1 or not p2 or p1[0] != p2[0]:
                continue
            records.append(
                {
                    "drug1": e1["text"],
                    "drug2": e2["text"],
                    "sentence": p1[1] if isinstance(p1[1], str) else "",
                    "relation": "none",
                    "doc_id": doc.get("document_id") or doc.get("id", ""),
                }
            )

    return records


def load_ddi(limit: int | None = None) -> pd.DataFrame:
    """Load DDI Corpus 2013 and annotate with per-pair severity.

    Returns
    -------
    DataFrame with columns:
        id, question, answer, severity, category, options, evidence, domain
    """
    if load_dataset is None:
        raise ImportError("Install the datasets package: pip install datasets")

    ds = load_dataset(
        "bigbio/ddi_corpus", "ddi_corpus_bigbio_kb", split="train", trust_remote_code=True
    )
    pairs = _iter_pairs_from_bigbio(ds)

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
        records.append(
            {
                "id": f"ddi_{i:06d}_{p['doc_id']}",
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
