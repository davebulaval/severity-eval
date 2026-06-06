"""LLM severity-from-output validation for FinanceBench.

For each (model, instance) cell where the model's prediction is wrong, send
the question + gold answer + the model's actual wrong prediction to a judge
LLM and ask for the severity of THIS specific error. Compare the resulting
severity distribution with the rule-based severity (which is derived from
the question's answer_type x metric_type matrix, not the model's output).

This addresses reviewer concern W3 (severity-from-question vs severity-from-output).

Usage:
    python experiments/validate_severity_output_llm.py --limit 20
    python experiments/validate_severity_output_llm.py  # all 12 models, full corpus

Output:
    results/output_severity_validation.json  -- per-(model, instance) labels
    results/output_severity_validation.csv   -- flat table
    Console: per-model kappa vs rule-based, plus pooled kappa.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from anthropic import Anthropic
from sklearn.metrics import cohen_kappa_score

# --- prompt -----------------------------------------------------------------

OUTPUT_SEVERITY_RUBRIC = """\
You are an expert financial analyst assessing the severity of an actual \
wrong answer produced by an AI system on a financial question.

Use this four-level scale calibrated to real-world financial impact:

**1. Negligible ($100)**
- The wrong answer is operationally indistinguishable from the correct one \
(rounding, paraphrase that preserves the financial meaning, immaterial digit \
substitution).

**2. Minor ($1,000)**
- The wrong answer is in the right order of magnitude on a derived ratio or \
percentage; would prompt a re-check but not a downstream decision.

**3. Major ($10,000)**
- The wrong answer differs by a factor that would change an operational \
metric (e.g., EBITDA, working capital, debt-to-equity) enough to trigger \
a different management decision.

**4. Critical ($100,000)**
- The wrong answer mis-states a core financial item (revenue, net income, \
total assets) by an amount that would shift a trading or credit decision; \
or flips the sign of a yes/no question about financial health; or moves \
a magnitude by an order of magnitude or more.

Respond in JSON only:
{
  "severity": "negligible" | "minor" | "major" | "critical",
  "justification": "One sentence explaining the severity of THIS specific \
wrong answer relative to the correct one."
}
"""


def build_prompt(question: str, gold: str, prediction: str) -> str:
    return (
        f"{OUTPUT_SEVERITY_RUBRIC}\n\n"
        f"---\n\n"
        f"**Question:** {question}\n\n"
        f"**Correct answer:** {gold}\n\n"
        f"**AI system's wrong answer:** {prediction}\n\n"
        f"Classify the severity of this specific wrong answer."
    )


def parse_llm_response(text: str) -> dict:
    """Extract {severity, justification} from the LLM response."""
    text = text.strip()
    # Strip code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        # last-resort: pull severity word
        sev_match = re.search(r"\"severity\"\s*:\s*\"(\w+)\"", text)
        sev = sev_match.group(1) if sev_match else "minor"
        return {"severity": sev, "justification": text[:200]}
    sev = d.get("severity", "minor").lower()
    if sev not in {"negligible", "minor", "major", "critical"}:
        sev = "minor"
    return {"severity": sev, "justification": d.get("justification", "")}


# --- main loop --------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/results"),
        help="Directory of {dataset}_{model}.json files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/output_severity_validation.json"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of wrong predictions per model to label "
        "(useful for smoke runs)",
    )
    parser.add_argument(
        "--model-id",
        default="claude-sonnet-4-5-20250929",
        help="Anthropic model id for the judge",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.1,
        help="Seconds to sleep between API calls (rate limit)",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = Anthropic()  # ANTHROPIC_API_KEY from env

    # Load all financebench_{model}.json files
    fb_files = sorted(glob.glob(str(args.results_dir / "financebench_*.json")))
    if not fb_files:
        sys.exit(f"No financebench_*.json files in {args.results_dir}")

    # Resume from existing output if present
    if args.output.exists():
        existing = json.loads(args.output.read_text())
        seen_keys = {(r["model"], r["id"]) for r in existing}
        print(
            f"Resuming: {len(seen_keys)} already-labelled (model, id) pairs", flush=True
        )
    else:
        existing = []
        seen_keys = set()

    n_calls = 0
    t0 = time.time()
    for fp in fb_files:
        model = Path(fp).stem.replace("financebench_", "")
        records = json.loads(Path(fp).read_text())
        wrong = [r for r in records if not r.get("correct", False)]
        if args.limit is not None:
            wrong = wrong[: args.limit]
        for r in wrong:
            key = (model, r["id"])
            if key in seen_keys:
                continue
            prompt = build_prompt(
                r.get("question", ""),
                str(r.get("answer", "")),
                str(r.get("prediction", "")),
            )
            try:
                resp = client.messages.create(
                    model=args.model_id,
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = "".join(blk.text for blk in resp.content if blk.type == "text")
                parsed = parse_llm_response(text)
            except Exception as e:  # noqa: BLE001
                print(f"  API error on {model}/{r['id']}: {e}", flush=True)
                parsed = {"severity": None, "justification": f"API error: {e}"}
            entry = {
                "model": model,
                "id": r["id"],
                "rule_severity": r.get("severity"),
                "llm_severity": parsed["severity"],
                "justification": parsed["justification"],
                "question": r.get("question", ""),
                "gold": str(r.get("answer", "")),
                "prediction": str(r.get("prediction", "")),
            }
            existing.append(entry)
            seen_keys.add(key)
            n_calls += 1
            if n_calls % 25 == 0:
                args.output.write_text(json.dumps(existing, indent=2))
                elapsed = time.time() - t0
                rate = n_calls / elapsed if elapsed > 0 else 0.0
                print(
                    f"  [{n_calls}] {model}/{r['id'][:30]} -> "
                    f"{parsed['severity']} (rate {rate:.1f}/s)",
                    flush=True,
                )
            if args.sleep > 0:
                time.sleep(args.sleep)

    args.output.write_text(json.dumps(existing, indent=2))
    print(f"\nTotal labelled this run: {n_calls}", flush=True)
    print(f"Output: {args.output}", flush=True)

    # Kappa computation
    valid = [
        e
        for e in existing
        if e["rule_severity"] is not None and e["llm_severity"] is not None
    ]
    if not valid:
        print("No valid pairs to compute kappa.", flush=True)
        return

    print("\n=== Inter-annotator agreement (rule vs LLM-from-output) ===")
    LEVELS = ["negligible", "minor", "major", "critical"]
    rule = [e["rule_severity"] for e in valid]
    llm = [e["llm_severity"] for e in valid]
    pooled = cohen_kappa_score(rule, llm, labels=LEVELS)
    pooled_adj = sum(
        1 for r, ll in zip(rule, llm) if abs(LEVELS.index(r) - LEVELS.index(ll)) <= 1
    ) / len(valid)
    print(
        f"Pooled (all models, n={len(valid)}): kappa={pooled:.3f}, adj agreement={pooled_adj:.1%}"
    )

    by_model = defaultdict(list)
    for e in valid:
        by_model[e["model"]].append((e["rule_severity"], e["llm_severity"]))
    for m, pairs in sorted(by_model.items()):
        if len(pairs) < 5:
            continue
        rs, ls = zip(*pairs)
        k = cohen_kappa_score(rs, ls, labels=LEVELS)
        adj = sum(
            1 for r, ll in zip(rs, ls) if abs(LEVELS.index(r) - LEVELS.index(ll)) <= 1
        ) / len(pairs)
        print(f"  {m:30}: n={len(pairs)}, kappa={k:.3f}, adj={adj:.1%}")

    # Flat CSV
    import csv

    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "id",
                "rule_severity",
                "llm_severity",
                "justification",
            ],
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(existing)
    print(f"CSV: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
