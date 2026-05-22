"""Produce a markdown summary of every (model, dataset) result JSON under
experiments/results/.

Reads all *_*.json files, aggregates accuracy + scoring methods + severity
profile per model, and prints a markdown report suitable for pasting into a
chat or the paper appendix.

Usage:
    python experiments/summarize_smoke.py
    python experiments/summarize_smoke.py --results-dir experiments/results --output report.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


KNOWN_DATASETS = (
    "financebench",
    "finqa",
    "tatqa",
    "medcalc",
    "medqa",
    "headqa",
    "cuad",
    "maud",
    "contractnli",
    "rag_insurance",
    "judgebert",
)


def _split_filename(stem: str) -> tuple[str, str] | None:
    """Parse `<dataset>_<model>[_<prompt_style>]` stems."""
    if stem.endswith("_standard"):
        stem = stem[: -len("_standard")]
    for ds in KNOWN_DATASETS:
        if stem.startswith(ds + "_"):
            return ds, stem[len(ds) + 1 :]
    return None


def load_results(results_dir: Path) -> dict:
    """Return {(model, dataset): list[record]}."""
    data: dict[tuple[str, str], list[dict]] = {}
    for f in sorted(results_dir.glob("*.json")):
        parsed = _split_filename(f.stem)
        if parsed is None:
            continue
        dataset, model = parsed
        try:
            data[(model, dataset)] = json.loads(f.read_text())
        except Exception as exc:
            print(f"[WARN] cannot read {f.name}: {exc}")
    return data


def summarize(data: dict) -> str:
    """Render a markdown report from the loaded JSONs."""
    by_model: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for (model, dataset), records in data.items():
        by_model[model][dataset] = records

    lines: list[str] = []
    lines.append("# Smoke test summary\n")
    lines.append(f"Total pairs: **{len(data)}**  ")
    lines.append(f"Models: **{len(by_model)}**  ")
    datasets_seen = {ds for _, ds in data}
    lines.append(f"Datasets: **{len(datasets_seen)}**\n")

    # --- Per-model summary -------------------------------------------------
    lines.append("## Per-model accuracy")
    lines.append("")
    lines.append("| Model | Datasets | n total | Correct | Avg acc | Domains touched |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for model in sorted(by_model):
        pairs = by_model[model]
        n_total = sum(len(r) for r in pairs.values())
        n_correct = sum(sum(1 for x in r if x.get("correct")) for r in pairs.values())
        avg_acc = (100 * n_correct / n_total) if n_total else 0.0
        domains = sorted({x.get("domain", "?") for r in pairs.values() for x in r})
        lines.append(
            f"| `{model}` | {len(pairs)} | {n_total} | {n_correct} | {avg_acc:.1f}% | {', '.join(domains)} |"
        )
    lines.append("")

    # --- Per-(model, dataset) matrix --------------------------------------
    lines.append("## Per-(model, dataset) accuracy matrix")
    lines.append("")
    datasets_order = [d for d in KNOWN_DATASETS if d in datasets_seen]
    header = "| Model | " + " | ".join(datasets_order) + " | Mean |"
    sep = "|---|" + ":---:|" * (len(datasets_order) + 1)
    lines.append(header)
    lines.append(sep)
    for model in sorted(by_model):
        row = [f"`{model}`"]
        accs = []
        for d in datasets_order:
            r = by_model[model].get(d)
            if not r:
                row.append("-")
                continue
            n = len(r)
            c = sum(1 for x in r if x.get("correct"))
            acc = 100 * c / n if n else 0
            accs.append(acc)
            row.append(f"{acc:.0f}%")
        row.append(f"**{sum(accs) / len(accs):.0f}%**" if accs else "-")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # --- Scoring method distribution per model ----------------------------
    lines.append("## Scoring methods (sanity check: avoid 'batch_error' / 'empty')")
    lines.append("")
    lines.append(
        "| Model | mcq | numeric | yes_no | exact | fuzzy_contains | fuzzy_words | no_match | empty | batch_error |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    method_cols = [
        "mcq",
        "numeric",
        "yes_no",
        "exact",
        "fuzzy_contains",
        "fuzzy_words",
        "no_match",
        "empty",
        "batch_error",
    ]
    for model in sorted(by_model):
        all_recs = [x for r in by_model[model].values() for x in r]
        counts = Counter(x.get("score_method", "?") for x in all_recs)
        row = [f"`{model}`"] + [str(counts.get(c, 0)) for c in method_cols]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # --- Severity-of-errors distribution per model ------------------------
    lines.append("## Error severity profile (counts among errors)")
    lines.append("")
    lines.append("| Model | negligible | minor | major | critical | total errors |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    sev_labels = ["negligible", "minor", "major", "critical"]
    for model in sorted(by_model):
        all_recs = [x for r in by_model[model].values() for x in r]
        errors = [x for x in all_recs if not x.get("correct")]
        sev_counts = Counter(x.get("severity", "?") for x in errors)
        row = [f"`{model}`"] + [str(sev_counts.get(s, 0)) for s in sev_labels]
        row.append(str(len(errors)))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # --- Anomalies / warnings ---------------------------------------------
    issues: list[str] = []
    for model in sorted(by_model):
        for ds in datasets_order:
            r = by_model[model].get(ds)
            if not r:
                continue
            counts = Counter(x.get("score_method", "?") for x in r)
            if counts.get("batch_error", 0):
                issues.append(
                    f"- `{model}` x `{ds}`: {counts['batch_error']} batch_error (pipeline crash)"
                )
            if counts.get("empty", 0) >= len(r) / 2:
                issues.append(
                    f"- `{model}` x `{ds}`: {counts['empty']}/{len(r)} empty predictions (output truncated?)"
                )
    if issues:
        lines.append("## Issues detected")
        lines.append("")
        lines.extend(issues)
        lines.append("")
    else:
        lines.append("## Issues detected")
        lines.append("")
        lines.append("None — every (model, dataset) produced real predictions.")
        lines.append("")

    # --- Datasets missing per model ---------------------------------------
    incomplete: list[str] = []
    for model in sorted(by_model):
        missing = [d for d in datasets_order if d not in by_model[model]]
        if missing:
            incomplete.append(f"- `{model}`: missing {', '.join(missing)}")
    if incomplete:
        lines.append("## Incomplete coverage")
        lines.append("")
        lines.extend(incomplete)
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("experiments/results"))
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the markdown report; default stdout only",
    )
    args = parser.parse_args()

    data = load_results(args.results_dir)
    if not data:
        print("[ABORT] no result JSONs found in", args.results_dir)
        raise SystemExit(1)

    report = summarize(data)
    print(report)
    if args.output:
        args.output.write_text(report)
        print(f"\n[OK] report saved to {args.output}", file=__import__("sys").stderr)


if __name__ == "__main__":
    main()
