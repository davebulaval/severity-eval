"""Load and annotate PrivacyQA dataset with severity levels.

PrivacyQA (Ravichander et al. 2019, "Question Answering for Privacy
Policies: Combining Computational and Legal Perspectives", EMNLP 2019):
1,750 questions about 35 mobile-app privacy policies, with each
(question, segment) pair annotated as Relevant / Irrelevant by a panel
of trained legal experts. The reading-comprehension task is to identify
which segments of a privacy policy address a user's natural-language
question.

Source: prefer HuggingFace mirror (alzoubi36/privacy_qa); fall back to
        local CSVs from the official GitHub release
        (AbhilashaRavichander/PrivacyQA_EMNLP) placed under
        ``dataset/privacyqa/policy_{train,test}_data.csv``.
License: research use (Ravichander et al. 2019).

Severity assignment
-------------------
Severity is the financial cost of an LLM error on this instance. The
gold-standard expert relevance label fixes the question type; we map the
QUESTION TEXT to one of the OPP-115 privacy-practice categories (Wilson
et al. 2016) using a keyword classifier built from the OPP-115 schema's
own terminology, then map the category to a regulatory-fine tier.

The mapping is derived from the OPP-115 schema, not from arbitrary
heuristics, and the loader documents the per-category fine ranges used
to calibrate the severity tier. This is methodologically intermediate
between (a) fully native expert-per-instance severity (e.g. CUAD clause
types) and (b) keyword-only inference (e.g. the dropped medqa/medcalc).
The Limitations section of the paper flags PrivacyQA as
"category-mapping" severity rather than "domain-expert-derived"
per-instance.

Cost vector: insurance ($100, $2k, $10k, $250k) -- regulatory-compliance
tier shared with the AMF/insurance scale. Critical-tier covers GDPR-
class data security violations; major covers third-party / retention;
minor / negligible cover access, choice, and generic clauses.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


# ----------------------------------------------------------------------
# OPP-115 -> severity. Calibrated against regulatory-fine schedules
# (GDPR Art. 83 tiers; CCPA; AMF). The exposures cited per tier are
# illustrative ranges from public enforcement data.
# ----------------------------------------------------------------------
CATEGORY_SEVERITY: dict[str, str] = {
    "Data Security": "critical",          # ex. GDPR 4% turnover, breach
    "Third Party Sharing/Collection": "major",  # consent / transfer fines
    "Data Retention": "major",            # storage limitation breaches
    "First Party Collection/Use": "minor",
    "User Choice/Control": "minor",
    "User Access, Edit and Deletion": "minor",
    "Policy Change": "minor",
    "Do Not Track": "minor",
    "International and Specific Audiences": "negligible",
    "Other": "negligible",
}


# Keywords drawn from the OPP-115 schema's category descriptors so the
# mapping reflects the source taxonomy rather than ad-hoc lexicon.
# Order matters: earlier rules win (Data Security checked before
# First Party Collection because security keywords are more specific).
_CATEGORY_RULES: list[tuple[str, list[str]]] = [
    (
        "Data Security",
        [
            "secur", "encrypt", "password", "credential", "hash", "breach",
            "leak", "vulnerab", "safeguard", "protect", "ssl", "tls",
        ],
    ),
    (
        "Third Party Sharing/Collection",
        [
            "third party", "third-party", "third parties", "advertis",
            "partner", "affiliate", "share", "sell", "transferred to",
            "transfer my", "with other companies", "external compan",
        ],
    ),
    (
        "Data Retention",
        [
            "retention", "retain", "how long", "store for", "keep my data",
            "kept for", "delete my", "deletion period", "remove my data",
        ],
    ),
    (
        "User Access, Edit and Deletion",
        [
            "access my", "edit my", "review my", "modify my", "correct my",
            "delete my account", "request my data", "see my data",
        ],
    ),
    (
        "User Choice/Control",
        [
            "opt-out", "opt out", "consent", "preference", "choice",
            "control", "do not", "opt in", "opt-in",
        ],
    ),
    (
        "Policy Change",
        [
            "change to", "policy update", "modif", "amend", "revise",
            "notification of change", "update this policy",
        ],
    ),
    (
        "Do Not Track",
        ["do not track", "dnt"],
    ),
    (
        "International and Specific Audiences",
        [
            "children", "minor", "under 13", "coppa", "outside the",
            "international", "eu user", "european", "california",
        ],
    ),
    (
        "First Party Collection/Use",
        [
            "collect", "gather", "obtain", "receive", "track", "monitor",
            "log", "record",
        ],
    ),
]


def classify_question_category(question: str) -> str:
    """Map a question to an OPP-115 category via the schema keyword rules.

    Falls back to ``Other`` if no rule fires.
    """
    q = (question or "").lower()
    for cat, kws in _CATEGORY_RULES:
        for kw in kws:
            if kw in q:
                return cat
    return "Other"


def classify_severity(question: str) -> tuple[str, str]:
    """Return (severity, opp115_category) for a PrivacyQA question."""
    cat = classify_question_category(question)
    return CATEGORY_SEVERITY.get(cat, "negligible"), cat


# Local fallback paths (relative to the project root).
_LOCAL_CSV_DIR = (
    Path(__file__).resolve().parents[2] / "dataset" / "privacyqa"
)


def _load_from_local() -> pd.DataFrame:
    """Read official CSVs (train + test) if present locally."""
    parts = []
    for split in ("train", "test"):
        p = _LOCAL_CSV_DIR / f"policy_{split}_data.csv"
        if not p.exists():
            continue
        # Original files are TSV-like with mixed quoting; try both.
        for sep in (",", "\t"):
            try:
                parts.append(pd.read_csv(p, sep=sep))
                break
            except Exception:
                continue
    if not parts:
        raise FileNotFoundError(
            f"PrivacyQA CSVs not found under {_LOCAL_CSV_DIR}. Download from "
            "https://github.com/AbhilashaRavichander/PrivacyQA_EMNLP or "
            "install via Hugging Face (alzoubi36/privacy_qa)."
        )
    return pd.concat(parts, ignore_index=True)


def _coerce_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names across the HF mirror and the original CSVs.

    Returns a frame with columns: ``query, segment, label`` (label is
    ``Y`` / ``N``).
    """
    rename: dict[str, str] = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("query", "question", "question_text"):
            rename[c] = "query"
        elif cl in ("segment", "text", "segment_text", "sentence"):
            rename[c] = "segment"
        elif cl in ("any_relevant", "label", "relevance", "is_relevant"):
            rename[c] = "label"
        elif cl in ("queryid", "query_id"):
            rename[c] = "query_id"
        elif cl in ("docid", "doc_id"):
            rename[c] = "doc_id"
        elif cl in ("segmentid", "segment_id"):
            rename[c] = "segment_id"
    out = df.rename(columns=rename)
    needed = {"query", "segment", "label"}
    missing = needed - set(out.columns)
    if missing:
        raise ValueError(
            f"PrivacyQA frame is missing columns {missing}; got {list(df.columns)}"
        )
    # Normalise label to Y/N
    def _norm(x):
        s = str(x).strip().lower()
        if s in ("y", "yes", "1", "true", "relevant"):
            return "Y"
        return "N"

    out["label"] = out["label"].map(_norm)
    return out


def load_privacyqa(limit: int | None = None) -> pd.DataFrame:
    """Load PrivacyQA pairs and annotate with severity.

    Returns
    -------
    DataFrame with columns:
        id, question, answer, severity, category, options, evidence, domain
    """
    # Prefer HuggingFace mirror; fall back to local CSV if unavailable.
    raw: pd.DataFrame | None = None
    if load_dataset is not None:
        for hf_name in ("alzoubi36/privacy_qa", "nateraw/privacyqa"):
            try:
                ds = load_dataset(hf_name, split="train")
                raw = ds.to_pandas()
                break
            except Exception:
                continue
    if raw is None:
        raw = _load_from_local()

    raw = _coerce_columns(raw)

    options = {"A": "Yes (relevant)", "B": "No (irrelevant)"}
    records: list[dict] = []
    # Use a stable row id : (doc_id, query_id, segment_id) tuple if
    # available, else the row index.
    for i, row in raw.iterrows():
        if limit is not None and len(records) >= limit:
            break
        query = (row.get("query") or "").strip()
        segment = (row.get("segment") or "").strip()
        if not query or not segment:
            continue
        gold_y = row.get("label") == "Y"
        gold_letter = "A" if gold_y else "B"
        severity, category = classify_severity(query)
        rid_parts = [
            str(row.get("doc_id", "")).strip(),
            str(row.get("query_id", "")).strip(),
            str(row.get("segment_id", i)).strip(),
        ]
        rid = "_".join(filter(None, rid_parts)) or f"row{i:06d}"
        records.append(
            {
                "id": f"privacyqa_{i:06d}_{re.sub(r'[^A-Za-z0-9_-]', '', rid)[:48]}",
                "question": (
                    f"Question about a privacy policy: {query}\n\n"
                    f"Is the following policy segment relevant to this "
                    f"question?\n\nSegment: {segment}"
                ),
                "answer": gold_letter,
                "answer_text": options[gold_letter],
                "severity": severity,
                "category": category,
                "options": dict(options),
                "evidence": segment,
                "domain": "insurance",
            }
        )

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = load_privacyqa(limit=10)
    print(f"Loaded {len(df)} rows")
    print(df[["id", "category", "severity", "answer"]].to_string())
    print("\nSeverity distribution:")
    print(df["severity"].value_counts())
