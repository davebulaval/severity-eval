"""Load and annotate ContractNLI dataset with severity levels.

ContractNLI: ~9,788 NLI examples (train/dev/test) on Non-Disclosure Agreements.
Source: kiddothe2b/contract-nli on HuggingFace (loaded from contract_nli.zip)
License: CC BY 4.0

Format: each example pairs a contract premise with one of 17 fixed hypotheses.
The task is to determine: entailment / neutral / contradiction.

Note: the HuggingFace repository uses a loading script that is no longer
supported by recent versions of the `datasets` library, so this loader fetches
the raw zip archive directly via huggingface_hub and reads the JSONL file
inside it.

Severity is assigned based on the hypothesis:
  - critical: return/destruction of confidential info, sharing with third
              parties, no reverse engineering
  - major:    limited use, non-disclosure, non-compete (no-solicit),
              survival of obligations
  - minor:    notice/disclosure requirements, permissible development,
              permissible copying
  - negligible: definitional / general provisions (right grants, identification
                requirements, verbally conveyed info, scope of CI definition,
                independent development, third-party acquisition)

Cost vector: Legal domain ($200, $2k, $20k, $200k).
"""

from __future__ import annotations

import json
import zipfile

import pandas as pd

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None


# ──────────────────────────────────────────────────────────────────────────────
# The 17 fixed hypotheses and their severity assignments.
# Keys are exact hypothesis strings (case-sensitive).
# ──────────────────────────────────────────────────────────────────────────────
HYPOTHESIS_SEVERITY: dict[str, str] = {
    # Critical — breach leads to immediate, high-exposure legal action
    "Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.": "critical",
    "Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).": "critical",
    "Receiving Party shall not reverse engineer any objects which embody Disclosing Party's Confidential Information.": "critical",
    # Major — core NDA obligations
    "Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement.": "major",
    "Receiving Party shall not disclose the fact that Agreement was agreed or negotiated.": "major",
    "Receiving Party shall not solicit some of Disclosing Party's representatives.": "major",
    "Some obligations of Agreement may survive termination of Agreement.": "major",
    "Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.": "major",
    # Minor — procedural / limited-risk provisions
    "Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information.": "minor",
    "Receiving Party may share some Confidential Information with some of Receiving Party's employees.": "minor",
    "Receiving Party may create a copy of some Confidential Information in some circumstances.": "minor",
    # Negligible — definitional / general / low-stakes
    "Agreement shall not grant Receiving Party any right to Confidential Information.": "negligible",
    "All Confidential Information shall be expressly identified by the Disclosing Party.": "negligible",
    "Confidential Information may include verbally conveyed information.": "negligible",
    "Confidential Information shall only include technical information.": "negligible",
    "Receiving Party may acquire information similar to Confidential Information from a third party.": "negligible",
    "Receiving Party may independently develop information similar to Confidential Information.": "negligible",
}

_REPO_ID = "kiddothe2b/contract-nli"
_FILENAME = "contract_nli.zip"
_JSONL_NAME = "contract_nli_v1.jsonl"


def _load_jsonl_from_hub() -> list[dict]:
    """Download and parse the JSONL file from the HuggingFace zip archive."""
    if hf_hub_download is None:
        raise ImportError("Install huggingface_hub: pip install huggingface-hub")

    zip_path = hf_hub_download(
        repo_id=_REPO_ID,
        filename=_FILENAME,
        repo_type="dataset",
    )

    with zipfile.ZipFile(zip_path, "r") as zf, zf.open(_JSONL_NAME) as f:
        raw = f.read().decode("utf-8")

    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def load_contractnli(limit: int | None = None) -> pd.DataFrame:
    """Load ContractNLI and annotate with severity.

    Loads all splits (train, dev, test) combined. The original split is
    preserved in a 'subset' column.

    Parameters
    ----------
    limit : int or None
        Max number of instances to load (combined across all subsets).

    Returns
    -------
    DataFrame with columns:
        id, question (hypothesis), answer (label), severity,
        document_id (premise hash for grouping), subset, domain
    """
    examples = _load_jsonl_from_hub()

    if limit is not None:
        examples = examples[:limit]

    records: list[dict] = []
    for i, row in enumerate(examples):
        hypothesis = row.get("hypothesis", "")
        label = row.get("label", "")
        premise = row.get("premise", "")
        subset = row.get("subset", "")

        severity = HYPOTHESIS_SEVERITY.get(hypothesis, "minor")

        # Use a short stable ID derived from the premise content so that
        # rows from the same contract can be grouped. We truncate to 60 chars
        # and make it filesystem-safe.
        doc_snippet = premise[:60].replace(" ", "_").replace("/", "-")

        records.append(
            {
                "id": f"contractnli_{i:05d}",
                "question": hypothesis,
                "answer": label,
                "severity": severity,
                "document_id": doc_snippet,
                "subset": subset,
                "domain": "legal",
            }
        )

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = load_contractnli(limit=5)
    print(df.to_string())
    print()
    print("Columns:", list(df.columns))
    print()
    df_full = load_contractnli()
    print(f"Total rows: {len(df_full)}")
    print("Severity counts:", df_full["severity"].value_counts().to_dict())
    print("Label counts:", df_full["answer"].value_counts().to_dict())
    print("Subset counts:", df_full["subset"].value_counts().to_dict())
