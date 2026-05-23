"""Tests for experiments.summarize_smoke.

The module is pure logic: it parses result filenames, loads JSON
records, and renders a markdown report. We exercise every section of
the report and every classification branch (issue detection, missing
coverage, scoring-method bucketing).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.summarize_smoke import (
    KNOWN_DATASETS,
    _split_filename,
    load_results,
    summarize,
)


# ----------------------------------------------------------------------
# _split_filename — disambiguates dataset/model in the stem
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "stem, expected",
    [
        ("financebench_o3", ("financebench", "o3")),
        ("medqa_claude-haiku", ("medqa", "claude-haiku")),
        # rag_insurance has an underscore; must match against the longer name
        ("rag_insurance_claude-haiku", ("rag_insurance", "claude-haiku")),
        # Standard suffix is stripped before the dataset match
        ("medqa_claude-haiku_standard", ("medqa", "claude-haiku")),
        ("rag_insurance_o3_standard", ("rag_insurance", "o3")),
        # Multi-segment model names with hyphens
        (
            "financebench_deepseek-r1-distill-70b",
            ("financebench", "deepseek-r1-distill-70b"),
        ),
        # Unknown dataset → None
        ("foobar_o3", None),
        ("nothing", None),
    ],
)
def test_split_filename(stem: str, expected: tuple[str, str] | None) -> None:
    assert _split_filename(stem) == expected


def test_known_datasets_contains_paper_set():
    """The hard-coded list must cover every dataset evaluated in the paper."""
    expected = {
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
    }
    assert set(KNOWN_DATASETS) == expected


# ----------------------------------------------------------------------
# load_results — reads the result jsons into a dict
# ----------------------------------------------------------------------


def _write_result(dir_: Path, dataset: str, model: str, records: list[dict]) -> None:
    (dir_ / f"{dataset}_{model}.json").write_text(json.dumps(records))


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    _write_result(
        tmp_path,
        "medqa",
        "modelA",
        [{"correct": True, "score_method": "mcq", "severity": "major"}],
    )
    _write_result(
        tmp_path,
        "financebench",
        "modelA",
        [
            {"correct": False, "score_method": "numeric", "severity": "critical"},
            {"correct": True, "score_method": "yes_no", "severity": "negligible"},
        ],
    )
    _write_result(
        tmp_path,
        "rag_insurance",
        "modelA",
        [{"correct": False, "score_method": "batch_error", "severity": "minor"}],
    )
    _write_result(
        tmp_path,
        "medqa",
        "modelB",
        [{"correct": True, "score_method": "mcq", "severity": "major"}],
    )
    # A junk filename that should be ignored
    (tmp_path / "unrelated.json").write_text(json.dumps([]))
    return tmp_path


def test_load_results_keys_are_model_dataset_tuples(results_dir: Path) -> None:
    data = load_results(results_dir)
    keys = set(data.keys())
    assert ("modelA", "medqa") in keys
    assert ("modelA", "financebench") in keys
    assert ("modelA", "rag_insurance") in keys
    assert ("modelB", "medqa") in keys
    # The unrelated file is filtered out
    assert not any("unrelated" in m for m, _ in keys)


def test_load_results_records_count(results_dir: Path) -> None:
    data = load_results(results_dir)
    assert len(data[("modelA", "medqa")]) == 1
    assert len(data[("modelA", "financebench")]) == 2


def test_load_results_skips_malformed_json(tmp_path: Path) -> None:
    (tmp_path / "medqa_bad.json").write_text("{not valid")
    (tmp_path / "medqa_good.json").write_text(json.dumps([{"correct": True}]))
    data = load_results(tmp_path)
    assert ("good", "medqa") in data
    assert ("bad", "medqa") not in data


# ----------------------------------------------------------------------
# summarize — full markdown report
# ----------------------------------------------------------------------


@pytest.fixture
def report(results_dir: Path) -> str:
    return summarize(load_results(results_dir))


def test_report_has_all_required_sections(report: str) -> None:
    for header in (
        "# Smoke test summary",
        "## Per-model accuracy",
        "## Per-(model, dataset) accuracy matrix",
        "## Scoring methods",
        "## Error severity profile",
        "## Issues detected",
    ):
        assert header in report, f"missing section header: {header}"


def test_report_counts_models_and_datasets(report: str) -> None:
    assert "Total pairs: **4**" in report
    assert "Models: **2**" in report
    # 3 distinct datasets touched: medqa, financebench, rag_insurance
    assert "Datasets: **3**" in report


def test_report_per_model_average_accuracy(report: str) -> None:
    """modelA: 2 correct out of 4 = 50%. modelB: 1/1 = 100%."""
    # Find the modelA row in the per-model table
    assert "`modelA` | 3 | 4 | 2 | 50.0%" in report
    assert "`modelB` | 1 | 1 | 1 | 100.0%" in report


def test_report_flags_batch_error(report: str) -> None:
    """The rag_insurance result has 1 batch_error in 1 record (100%)."""
    assert "modelA` x `rag_insurance`" in report
    assert "batch_error" in report


def test_report_marks_incomplete_coverage(report: str) -> None:
    # modelA covered 3 of 11 datasets; should appear in incomplete section
    assert "## Incomplete coverage" in report
    assert "missing" in report


def test_report_no_issues_when_clean(tmp_path: Path) -> None:
    """When no result has batch_error or majority-empty, issues section says 'None'."""
    _write_result(
        tmp_path,
        "medqa",
        "modelA",
        [
            {"correct": True, "score_method": "mcq", "severity": "major"},
            {"correct": False, "score_method": "mcq", "severity": "minor"},
        ],
    )
    report = summarize(load_results(tmp_path))
    assert "None — every (model, dataset) produced real predictions." in report


def test_report_flags_majority_empty(tmp_path: Path) -> None:
    """If >=50% predictions are empty, flag as truncation issue."""
    _write_result(
        tmp_path,
        "medqa",
        "modelA",
        [
            {"correct": False, "score_method": "empty", "severity": "major"},
            {"correct": False, "score_method": "empty", "severity": "major"},
            {"correct": True, "score_method": "mcq", "severity": "minor"},
        ],
    )
    report = summarize(load_results(tmp_path))
    assert "empty predictions" in report
    assert "modelA` x `medqa`" in report


def test_report_scoring_method_counts_aggregated(report: str) -> None:
    """The score_methods row sums every record across a model's datasets."""
    # modelA: 1 mcq + 1 numeric + 1 yes_no + 1 batch_error = 4 records.
    # Find the modelA row in the scoring-methods table by looking for the
    # backtick-quoted name followed by the method counts.
    lines = report.splitlines()
    score_section_start = next(
        i for i, ln in enumerate(lines) if ln.startswith("## Scoring methods")
    )
    score_section = "\n".join(lines[score_section_start : score_section_start + 12])
    # Find modelA row
    modela_row = [
        ln for ln in score_section.splitlines() if ln.startswith("| `modelA`")
    ][0]
    cells = [c.strip() for c in modela_row.split("|")]
    # Columns: model, mcq, numeric, yes_no, exact, fuzzy_contains, fuzzy_words, no_match, empty, batch_error
    assert cells[2] == "1"  # mcq
    assert cells[3] == "1"  # numeric
    assert cells[4] == "1"  # yes_no
    assert cells[10] == "1"  # batch_error


def test_report_severity_profile_among_errors(tmp_path: Path) -> None:
    """The severity table counts only errors (correct=False)."""
    _write_result(
        tmp_path,
        "medqa",
        "modelA",
        [
            {"correct": True, "score_method": "mcq", "severity": "critical"},
            {"correct": False, "score_method": "mcq", "severity": "critical"},
            {"correct": False, "score_method": "mcq", "severity": "minor"},
        ],
    )
    report = summarize(load_results(tmp_path))
    # Errors: 1 critical, 1 minor — the correct critical is excluded
    lines = report.splitlines()
    sev_start = next(i for i, ln in enumerate(lines) if "Error severity profile" in ln)
    sev_section = "\n".join(lines[sev_start : sev_start + 10])
    row = [ln for ln in sev_section.splitlines() if ln.startswith("| `modelA`")][0]
    cells = [c.strip() for c in row.split("|")]
    # columns: model, negligible, minor, major, critical, total errors
    assert cells[2] == "0"  # negligible
    assert cells[3] == "1"  # minor
    assert cells[4] == "0"  # major
    assert cells[5] == "1"  # critical
    assert cells[6] == "2"  # total errors
