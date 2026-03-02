"""LLM-based validation of severity annotations.

Sends each (question, answer) to a judge LLM with the severity rubric,
collects independent severity labels + justifications, and computes
inter-annotator agreement (Cohen's kappa) with the rule-based system.

Usage:
    python experiments/validate_severity_llm.py --dataset financebench --limit 10
    python experiments/validate_severity_llm.py --dataset financebench
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from anthropic import Anthropic
from sklearn.metrics import cohen_kappa_score, confusion_matrix

SEVERITY_RUBRIC = """\
You are an expert financial analyst tasked with assessing the severity of \
potential errors in financial question-answering.

Given a question about a company's SEC filing and its correct answer, \
classify how severe it would be if an AI system got this answer WRONG.

Use this four-level scale calibrated to real-world financial impact:

**1. Negligible ($100)**
- Qualitative descriptions, company overviews, filing metadata
- Non-numerical answers about products, services, key agenda items
- Errors here have no direct financial impact on decisions

**2. Minor ($1,000)**
- Ratios, margins, percentages, growth rates, per-share values (EPS)
- Derived metrics that are bounded in magnitude
- Errors correspond to "little r" restatement territory (immaterial corrections)

**3. Major ($10,000)**
- Operational metrics in absolute dollars: EBITDA, cash flow, capex, debt, \
working capital
- Yes/no questions about financial health with numerical implications
- Errors at this level could trigger SOX Section 906 individual penalties ($10k)

**4. Critical ($100,000)**
- Core financial statement items in millions/billions: revenue, net income, \
total assets, total equity
- Errors here are "Big R" restatement territory (SEC enforcement, FINRA median \
fine $125k, class action risk)

Respond in JSON format:
{
  "severity": "negligible" | "minor" | "major" | "critical",
  "justification": "One sentence explaining why."
}
"""

DATASETS = {
    "financebench": "experiments.datasets.load_financebench:load_financebench",
}


def build_prompt(question: str, answer: str) -> str:
    """Build the judge prompt for a single instance."""
    return (
        f"{SEVERITY_RUBRIC}\n\n"
        f"---\n\n"
        f"**Question:** {question}\n\n"
        f"**Correct answer:** {answer}\n\n"
        f"Classify the severity of a wrong answer to this question."
    )


def parse_llm_response(text: str) -> dict:
    """Extract severity and justification from LLM response."""
    text = text.strip()
    # Try to parse as JSON
    # Handle markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(text)
        severity = data.get("severity", "").lower().strip()
        justification = data.get("justification", "")
        if severity in ("negligible", "minor", "major", "critical"):
            return {"llm_severity": severity, "llm_justification": justification}
    except json.JSONDecodeError:
        pass

    # Fallback: look for severity keyword in text
    text_lower = text.lower()
    for level in ("critical", "major", "minor", "negligible"):
        if level in text_lower:
            return {"llm_severity": level, "llm_justification": text}

    return {"llm_severity": "unknown", "llm_justification": text}


def validate_with_llm(
    df: pd.DataFrame,
    output_path: Path,
    model: str = "claude-sonnet-4-20250514",
    delay: float = 0.5,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Run LLM validation on a severity-annotated dataset.

    Parameters
    ----------
    df : DataFrame
        Dataset with 'question', 'answer', 'severity' columns.
    output_path : Path
        Path to save detailed results.
    model : str
        Anthropic model ID.
    delay : float
        Delay between API calls.
    max_retries : int
        Max retries per query.

    Returns
    -------
    DataFrame with added 'llm_severity' and 'llm_justification' columns.
    """
    client = Anthropic()
    results = []

    for i, row in df.iterrows():
        prompt = build_prompt(row["question"], row["answer"])

        response_text = ""
        for attempt in range(max_retries):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=256,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )
                response_text = response.content[0].text
                break
            except Exception as e:
                print(f"  Retry {attempt + 1}/{max_retries} for {row['id']}: {e}")
                time.sleep(delay * (attempt + 1))

        parsed = parse_llm_response(response_text)

        results.append(
            {
                **row.to_dict(),
                "llm_severity": parsed["llm_severity"],
                "llm_justification": parsed["llm_justification"],
                "agree": row["severity"] == parsed["llm_severity"],
            }
        )

        if delay > 0:
            time.sleep(delay)

        if (i + 1) % 25 == 0:
            print(f"  Processed {i + 1}/{len(df)} instances")

    results_df = pd.DataFrame(results)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_json(output_path, orient="records", indent=2)
    print(f"  Saved {len(results_df)} results to {output_path}")

    return results_df


def compute_agreement(df: pd.DataFrame) -> dict:
    """Compute inter-annotator agreement metrics.

    Parameters
    ----------
    df : DataFrame
        Must have 'severity' (rule-based) and 'llm_severity' (LLM) columns.

    Returns
    -------
    dict with kappa, accuracy, confusion matrix, and per-level stats.
    """
    labels = ["negligible", "minor", "major", "critical"]

    # Filter out unknowns
    valid = df[df["llm_severity"].isin(labels)].copy()
    n_unknown = len(df) - len(valid)

    rule_labels = valid["severity"].values
    llm_labels = valid["llm_severity"].values

    # Cohen's kappa
    kappa = cohen_kappa_score(rule_labels, llm_labels, labels=labels)

    # Exact agreement
    accuracy = (rule_labels == llm_labels).mean()

    # Adjacent agreement (±1 level)
    level_map = {lbl: i for i, lbl in enumerate(labels)}
    rule_idx = np.array([level_map[lbl] for lbl in rule_labels])
    llm_idx = np.array([level_map[lbl] for lbl in llm_labels])
    adjacent_accuracy = (np.abs(rule_idx - llm_idx) <= 1).mean()

    # Confusion matrix
    cm = confusion_matrix(rule_labels, llm_labels, labels=labels)

    # Per-level agreement
    per_level = {}
    for level in labels:
        mask = rule_labels == level
        if mask.sum() > 0:
            per_level[level] = {
                "n": int(mask.sum()),
                "exact_agree": float((llm_labels[mask] == level).mean()),
            }

    # Disagreement details
    disagree = valid[valid["severity"] != valid["llm_severity"]]
    disagreements = []
    for _, row in disagree.iterrows():
        disagreements.append(
            {
                "id": row["id"],
                "question": row["question"][:100],
                "answer": row["answer"][:80],
                "rule_severity": row["severity"],
                "llm_severity": row["llm_severity"],
                "llm_justification": row["llm_justification"],
                "answer_type": row.get("answer_type", ""),
                "metric_type": row.get("metric_type", ""),
            }
        )

    return {
        "n_total": len(df),
        "n_valid": len(valid),
        "n_unknown": n_unknown,
        "kappa": float(kappa),
        "exact_accuracy": float(accuracy),
        "adjacent_accuracy": float(adjacent_accuracy),
        "confusion_matrix": cm.tolist(),
        "confusion_labels": labels,
        "per_level": per_level,
        "disagreements": disagreements,
    }


def print_report(agreement: dict):
    """Print a formatted agreement report."""
    print("\n" + "=" * 60)
    print("INTER-ANNOTATOR AGREEMENT REPORT")
    print("=" * 60)

    print(
        f"\nInstances:  {agreement['n_total']} total, "
        f"{agreement['n_valid']} valid, "
        f"{agreement['n_unknown']} unknown/unparsed"
    )

    print(f"\nCohen's kappa:       {agreement['kappa']:.3f}")
    print(f"Exact agreement:     {agreement['exact_accuracy']:.1%}")
    print(f"Adjacent agreement:  {agreement['adjacent_accuracy']:.1%}")

    print("\nConfusion matrix (rows=rule, cols=LLM):")
    labels = agreement["confusion_labels"]
    header = "            " + "  ".join(f"{lbl[:4]:>6}" for lbl in labels)
    print(header)
    cm = np.array(agreement["confusion_matrix"])
    for i, label in enumerate(labels):
        row = "  ".join(f"{v:>6d}" for v in cm[i])
        print(f"  {label:>10}  {row}")

    print("\nPer-level exact agreement:")
    for level, stats in agreement["per_level"].items():
        print(f"  {level:>10}: {stats['exact_agree']:.1%} ({stats['n']} instances)")

    n_disagree = len(agreement["disagreements"])
    print(f"\nDisagreements: {n_disagree}")
    if n_disagree > 0:
        print("\nSample disagreements:")
        for d in agreement["disagreements"][:10]:
            print(f"  [{d['id']}] rule={d['rule_severity']} vs llm={d['llm_severity']}")
            print(f"    Q: {d['question']}")
            print(f"    A: {d['answer']}")
            print(f"    LLM: {d['llm_justification'][:120]}")
            print()


def main():
    parser = argparse.ArgumentParser(description="Validate severity annotations with LLM judge")
    parser.add_argument(
        "--dataset",
        default="financebench",
        choices=list(DATASETS.keys()),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Anthropic model ID for judge",
    )
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        default="experiments/results/validation",
        help="Output directory",
    )
    args = parser.parse_args()

    # Load dataset
    module_path, func_name = DATASETS[args.dataset].rsplit(":", 1)
    import importlib

    module = importlib.import_module(module_path)
    load_fn = getattr(module, func_name)
    df = load_fn(limit=args.limit)
    print(f"Loaded {len(df)} instances from {args.dataset}")
    print(f"Rule-based distribution:\n{df['severity'].value_counts().to_string()}\n")

    # Run LLM validation
    output_dir = Path(args.output_dir)
    output_path = output_dir / f"{args.dataset}_llm_validation.json"

    if output_path.exists():
        print(f"Loading existing results from {output_path}")
        results_df = pd.read_json(output_path)
    else:
        print(f"Running LLM validation with {args.model}...")
        results_df = validate_with_llm(
            df,
            output_path,
            model=args.model,
            delay=args.delay,
        )

    # Compute agreement
    agreement = compute_agreement(results_df)

    # Save agreement report
    report_path = output_dir / f"{args.dataset}_agreement_report.json"
    with open(report_path, "w") as f:
        json.dump(agreement, f, indent=2)
    print(f"Agreement report saved to {report_path}")

    # Print report
    print_report(agreement)


if __name__ == "__main__":
    main()
