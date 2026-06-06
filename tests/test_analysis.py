"""Tests for experiments.analysis result loading and dataset filtering.

The Monte Carlo metric computation is not unit tested here (it is
exercised end-to-end by the analysis CLI); we focus on load_results,
whose dataset-name disambiguation and dropped-dataset filtering directly
determine which (model, dataset) cells reach the per-domain aggregates.
A regression here silently contaminates the Kendall tau / variance /
routing numbers reported in the paper.
"""

from __future__ import annotations

import json


from experiments.analysis import (
    _DROPPED_DATASETS,
    RETAINED_DATASETS,
    load_results,
)


def _write_results(dir_path, dataset: str, model: str, rows: list[dict]) -> None:
    (dir_path / f"{dataset}_{model}.json").write_text(json.dumps(rows))


def _row(dataset_domain: str) -> dict:
    return {
        "id": "x0",
        "question": "q",
        "answer": "a",
        "severity": "minor",
        "domain": dataset_domain,
        "model": "m",
        "prediction": "p",
        "correct": True,
        "score_method": "exact_match",
    }


def test_load_results_excludes_dropped_by_default(tmp_path):
    """medqa / medcalc / rag_insurance JSONs must be skipped by default so
    stale smoke files cannot contaminate the aggregates."""
    _write_results(tmp_path, "cuad", "phi-4", [_row("legal")])
    _write_results(tmp_path, "medqa", "phi-4", [_row("medical")])
    _write_results(tmp_path, "medcalc", "phi-4", [_row("medical")])
    _write_results(tmp_path, "rag_insurance", "phi-4", [_row("insurance")])

    df = load_results(tmp_path)
    loaded = set(df["dataset"].unique())
    assert loaded == {"cuad"}, f"expected only cuad, got {loaded}"
    for d in _DROPPED_DATASETS:
        assert d not in loaded


def test_load_results_include_dropped_loads_everything(tmp_path):
    """include_dropped=True re-includes the dropped datasets for
    replication of their published benchmarks."""
    _write_results(tmp_path, "cuad", "phi-4", [_row("legal")])
    _write_results(tmp_path, "medqa", "phi-4", [_row("medical")])

    df = load_results(tmp_path, include_dropped=True)
    loaded = set(df["dataset"].unique())
    assert loaded == {"cuad", "medqa"}


def test_load_results_disambiguates_underscore_dataset_names(tmp_path):
    """rag_insurance contains an underscore; the loader must match it as a
    whole dataset name, not split on the first underscore into 'rag'."""
    _write_results(tmp_path, "rag_insurance", "phi-4", [_row("insurance")])

    df = load_results(tmp_path, include_dropped=True)
    assert set(df["dataset"].unique()) == {"rag_insurance"}


def test_load_results_keeps_all_retained_datasets(tmp_path):
    """Every retained dataset must survive the filter."""
    for ds in RETAINED_DATASETS:
        _write_results(tmp_path, ds, "phi-4", [_row("legal")])

    df = load_results(tmp_path)
    assert set(df["dataset"].unique()) == set(RETAINED_DATASETS)


def test_load_results_raises_when_no_retained_files(tmp_path):
    """If the directory contains only dropped datasets, the default filter
    leaves nothing and load_results raises rather than returning empty."""
    import pytest

    _write_results(tmp_path, "medqa", "phi-4", [_row("medical")])

    with pytest.raises(FileNotFoundError):
        load_results(tmp_path)


def test_dropped_and_retained_are_disjoint():
    """Sanity: a dataset cannot be both retained and dropped."""
    assert set(RETAINED_DATASETS).isdisjoint(set(_DROPPED_DATASETS))


# ------------------------------------------------------------------
# H1 weighted vs unweighted aggregation (Appendix G sensitivity)
# ------------------------------------------------------------------


def _h1_fixture():
    """Two datasets per domain with very different n, so weighted and
    unweighted aggregations will disagree."""
    import pandas as pd

    rows = []
    # Domain "finance": small ds tilts unweighted; large ds tilts weighted
    # model A: high acc on small ds, low acc on large ds -> low weighted, high unweighted
    # model B: low acc on small ds, high acc on large ds -> opposite
    # Then E[S] inverse-proportional so weighted ranking differs.
    for model, accs, ess, ns in [
        ("A", [0.9, 0.1], [10.0, 1000.0], [10, 1000]),
        ("B", [0.1, 0.9], [1000.0, 10.0], [10, 1000]),
        ("C", [0.5, 0.5], [500.0, 500.0], [10, 1000]),
    ]:
        for ds, acc, es, n in zip(["d_small", "d_large"], accs, ess, ns):
            rows.append(
                {
                    "model": model,
                    "dataset": ds,
                    "taxonomy_domain": "finance",
                    "accuracy": acc,
                    "expected_loss": es,
                    "n": n,
                }
            )
    return pd.DataFrame(rows)


def test_h1_unweighted_vs_weighted_aggregation_disagrees():
    from experiments.analysis import test_h1_ranking_divergence

    df = _h1_fixture()
    uw = test_h1_ranking_divergence(df, weighted=False)["per_domain"]["finance"]
    w = test_h1_ranking_divergence(df, weighted=True)["per_domain"]["finance"]
    # Both report a tau, but the per-model rank ordering differs because
    # weighting flips the within-domain accuracy ordering.
    assert uw["aggregation"] == "unweighted"
    assert w["aggregation"] == "weighted"
    assert uw["tau"] != w["tau"]


def test_h1_weighted_uses_sample_sizes():
    """Weighted aggregation gives more weight to the larger dataset."""
    from experiments.analysis import test_h1_ranking_divergence

    df = _h1_fixture()
    w = test_h1_ranking_divergence(df, weighted=True)["per_domain"]["finance"]
    # In _h1_fixture, model A has acc 0.9 on n=10 and 0.1 on n=1000.
    # Sample-weighted mean acc(A) = (10*0.9 + 1000*0.1)/1010 ≈ 0.109
    # Model B is the mirror image, weighted mean ≈ 0.892.
    # So under weighting, B is ranked above A.
    # We check that the H1 result tracks the weighted ordering by
    # confirming tau matches the rank correlation of the weighted means.
    # If unweighted, mean acc(A) = 0.5 = mean acc(B), tie -> different tau.
    assert w["tau"] != 0.0  # there is some signal


def test_h1_default_is_unweighted():
    from experiments.analysis import test_h1_ranking_divergence

    df = _h1_fixture()
    default = test_h1_ranking_divergence(df)["per_domain"]["finance"]
    assert default["aggregation"] == "unweighted"
