"""Tests for severity_eval.taxonomy."""

from __future__ import annotations

import numpy as np
import pytest

from severity_eval.taxonomy import (
    SEVERITY_LABELS,
    Taxonomy,
    get_taxonomy,
    list_domains,
    register_taxonomy,
    severity_label_to_index,
)


def test_severity_labels_canonical():
    """The default severity ordering is the one assumed everywhere."""
    assert SEVERITY_LABELS == ["negligible", "minor", "major", "critical"]


def test_get_finance_costs():
    tax = get_taxonomy("finance")
    np.testing.assert_array_equal(tax.cost_levels, [100, 1_000, 10_000, 100_000])


def test_get_medical_costs_match_paper_table():
    """Cost vectors must match Table 1 of the paper."""
    np.testing.assert_array_equal(
        get_taxonomy("medical").cost_levels, [500, 5_000, 50_000, 500_000]
    )
    np.testing.assert_array_equal(
        get_taxonomy("legal").cost_levels, [200, 2_000, 20_000, 200_000]
    )
    np.testing.assert_array_equal(
        get_taxonomy("insurance").cost_levels, [100, 2_000, 10_000, 250_000]
    )


def test_get_taxonomy_case_insensitive():
    assert get_taxonomy("Finance").domain == "finance"
    assert get_taxonomy("LEGAL").domain == "legal"


def test_get_taxonomy_aliases():
    """Dataset-domain aliases collapse to the canonical taxonomy."""
    assert get_taxonomy("legal_nli").domain == "legal"
    assert get_taxonomy("legal_simplification").domain == "legal"
    assert get_taxonomy("law").domain == "legal"
    assert get_taxonomy("med").domain == "medical"
    assert get_taxonomy("healthcare").domain == "medical"
    assert get_taxonomy("rag_insurance").domain == "insurance"
    assert get_taxonomy("financial").domain == "finance"


def test_unknown_taxonomy_raises_keyerror():
    with pytest.raises(KeyError, match="Unknown taxonomy"):
        get_taxonomy("astrology")


def test_list_domains_returns_canonical_names():
    names = list_domains()
    assert set(names) >= {"finance", "medical", "legal", "insurance"}


def test_taxonomy_index():
    tax = get_taxonomy("finance")
    assert tax.index("negligible") == 0
    assert tax.index("critical") == len(tax.labels) - 1


def test_taxonomy_index_unknown_label_raises():
    tax = get_taxonomy("finance")
    with pytest.raises(ValueError, match="Unknown severity label"):
        tax.index("catastrophic")


def test_severity_label_to_index():
    assert severity_label_to_index("negligible") == 0
    assert severity_label_to_index("minor") == 1
    assert severity_label_to_index("major") == 2
    assert severity_label_to_index("critical") == 3


def test_severity_label_to_index_case_insensitive():
    assert severity_label_to_index("CRITICAL") == 3
    assert severity_label_to_index("  Major  ") == 2


def test_severity_label_to_index_unknown_raises():
    with pytest.raises(ValueError, match="Unknown severity label"):
        severity_label_to_index("catastrophic")


def test_register_custom_taxonomy_and_cleanup():
    """Custom taxonomies round-trip through get_taxonomy."""
    tax = Taxonomy(
        domain="code-review",
        labels=tuple(SEVERITY_LABELS),
        cost_levels=np.array([10.0, 100.0, 1_000.0, 10_000.0]),
        description="Test taxonomy",
    )
    register_taxonomy(tax)
    try:
        recovered = get_taxonomy("code-review")
        np.testing.assert_array_equal(recovered.cost_levels, tax.cost_levels)
        assert recovered.description == "Test taxonomy"
    finally:
        # Manual cleanup: register_taxonomy mutates module state.
        from severity_eval import taxonomy as tx

        tx._TAXONOMIES.pop("code-review", None)


def test_register_taxonomy_rejects_duplicate_without_overwrite():
    tax = Taxonomy(
        domain="finance",
        labels=tuple(SEVERITY_LABELS),
        cost_levels=np.array([1.0, 2.0, 3.0, 4.0]),
    )
    with pytest.raises(KeyError, match="already registered"):
        register_taxonomy(tax)


def test_register_taxonomy_overwrite_then_restore():
    """overwrite=True replaces the existing taxonomy; restore after test."""
    from severity_eval import taxonomy as tx

    original = tx._TAXONOMIES["finance"]
    custom = Taxonomy(
        domain="finance",
        labels=tuple(SEVERITY_LABELS),
        cost_levels=np.array([1.0, 2.0, 3.0, 4.0]),
    )
    register_taxonomy(custom, overwrite=True)
    try:
        np.testing.assert_array_equal(
            get_taxonomy("finance").cost_levels, [1.0, 2.0, 3.0, 4.0]
        )
    finally:
        tx._TAXONOMIES["finance"] = original


def test_taxonomy_costs_strictly_increasing():
    """Costs must be strictly monotone; otherwise routing thresholds break."""
    for name in ["finance", "medical", "legal", "insurance"]:
        costs = get_taxonomy(name).cost_levels
        assert (np.diff(costs) > 0).all(), f"{name} costs not increasing: {costs}"


def test_taxonomy_label_count_matches_costs():
    """Length of labels must match the cost vector."""
    for name in list_domains():
        tax = get_taxonomy(name)
        assert len(tax.labels) == len(tax.cost_levels)
