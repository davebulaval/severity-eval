"""Generate trivial-baseline predictions (random and majority) for every
dataset already present in experiments/results/.

For each (dataset, instance) row in an existing results JSON we reproduce the
exact metadata (id, question, answer, severity, options, ...) but replace the
model prediction by:

- random: uniform draw from the option set (MCQ) or empty string (open-ended).
- majority: the most-frequent gold answer across the dataset.

Each baseline is then scored by the same score_prediction cascade used for
the LLMs, so the resulting JSON drops into the existing analysis pipeline as
two additional models named 'random' and 'majority'.

Usage:
    python -m experiments.baseline_predict --seed 42

Writes experiments/results/{dataset}_random.json and
experiments/results/{dataset}_majority.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path

from experiments.evaluate_models import score_prediction


def _gold_majority(rows: list[dict]) -> str:
    """Most frequent gold answer in the dataset."""
    counts = Counter(str(r.get("answer", "")) for r in rows)
    answer, _ = counts.most_common(1)[0]
    return answer


def _random_prediction(row: dict, rng: random.Random) -> str:
    """Generate a uniform random prediction.

    For MCQ rows where ``options`` is a non-empty dict, pick a random key and
    return ``"{key}. {value}"`` (matches the format the LLM scorer expects).
    For everything else, return the empty string so the scorer falls through
    to 'empty' (always-incorrect).
    """
    options = row.get("options")
    if isinstance(options, dict) and options:
        key = rng.choice(sorted(options.keys()))
        return f"{key}. {options[key]}"
    return ""


def _build_baseline(
    rows: list[dict], baseline: str, rng: random.Random
) -> list[dict]:
    out = []
    majority = _gold_majority(rows) if baseline == "majority" else None
    for r in rows:
        new = deepcopy(r)
        if baseline == "random":
            pred = _random_prediction(r, rng)
        else:  # majority
            pred = majority or ""
        scored = score_prediction(
            pred,
            str(r.get("answer", "")),
            options=r.get("options") if isinstance(r.get("options"), dict) else None,
        )
        new["model"] = baseline
        new["prediction"] = pred
        new["correct"] = bool(scored["correct"])
        new["score_method"] = scored["score_method"]
        out.append(new)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/results"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Pick one source-of-truth file per dataset (the first model alphabetically
    # whose file exists). We only need the instance metadata; predictions are
    # overwritten.
    datasets: set[str] = set()
    for fp in sorted(glob.glob(str(args.results_dir / "*.json"))):
        stem = Path(fp).stem
        # parse <dataset>_<model>: dataset name may contain underscores
        # heuristic: dataset is everything before the last underscore-separated
        # known-model token. Simpler: drop the model suffix if it matches a
        # standard pool; otherwise treat the longest prefix as dataset.
        # We use a hard-coded dataset list from the paper.
        for ds in [
            "contractnli",
            "cuad",
            "ddi",
            "financebench",
            "finqa",
            "headqa",
            "judgebert",
            "maud",
            "medmcqa",
            "privacyqa",
            "tatqa",
        ]:
            if stem.startswith(ds + "_"):
                datasets.add(ds)
                break

    print(f"Datasets found: {sorted(datasets)}")

    for ds in sorted(datasets):
        # Use the first available file for this dataset as template.
        files = sorted(glob.glob(str(args.results_dir / f"{ds}_*.json")))
        # Skip any pre-existing baseline file when selecting the template.
        files = [f for f in files if not f.endswith("_random.json")]
        files = [f for f in files if not f.endswith("_majority.json")]
        if not files:
            print(f"  {ds}: no template found")
            continue
        template_rows = json.loads(Path(files[0]).read_text())
        for baseline in ("random", "majority"):
            rng = random.Random(args.seed)
            out_rows = _build_baseline(template_rows, baseline, rng)
            out_path = args.results_dir / f"{ds}_{baseline}.json"
            out_path.write_text(json.dumps(out_rows, indent=2))
            n_correct = sum(1 for r in out_rows if r["correct"])
            acc = n_correct / len(out_rows) if out_rows else 0.0
            print(
                f"  {ds:14}/{baseline:9}: n={len(out_rows):>6}, "
                f"acc={acc:.3f}, file={out_path.name}"
            )


if __name__ == "__main__":
    main()
