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
