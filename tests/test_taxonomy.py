"""Tests for taxonomy module."""

import numpy as np
import pytest

from severity_eval.taxonomy import (
    DOMAINS,
    get_taxonomy,
    list_domains,
    register_taxonomy,
    severity_label_to_index,
)


def test_all_domains_exist():
    """All 5 domains should be accessible."""
    assert len(DOMAINS) == 5
    for domain in DOMAINS:
        tax = get_taxonomy(domain)
        assert tax.domain == domain


def test_each_taxonomy_has_4_levels():
    """Each domain taxonomy should have exactly 4 cost levels."""
    for domain in DOMAINS:
        tax = get_taxonomy(domain)
        assert len(tax.cost_levels) == 4
        assert len(tax.labels) == 4


def test_cost_levels_are_increasing():
    """Cost levels should be strictly increasing within each domain."""
    for domain in DOMAINS:
        tax = get_taxonomy(domain)
        for i in range(len(tax.cost_levels) - 1):
            assert tax.cost_levels[i] < tax.cost_levels[i + 1]


def test_finance_cost_levels():
    """Finance domain should have specific cost levels."""
    tax = get_taxonomy("finance")
    np.testing.assert_array_equal(tax.cost_levels, [100, 1000, 10000, 100000])


def test_medical_cost_levels():
    """Medical domain should have highest costs (safety-critical)."""
    tax = get_taxonomy("medical")
    np.testing.assert_array_equal(tax.cost_levels, [500, 5000, 50000, 500000])


def test_validate_profile():
    """Profile validation should work correctly."""
    tax = get_taxonomy("finance")
    assert tax.validate_profile(np.array([0.4, 0.3, 0.2, 0.1]))
    assert not tax.validate_profile(np.array([0.5, 0.3, 0.2, 0.1]))  # doesn't sum to 1
    assert not tax.validate_profile(np.array([0.5, 0.5]))  # wrong length


def test_severity_label_to_index():
    """Label to index conversion should work."""
    assert severity_label_to_index("negligible") == 0
    assert severity_label_to_index("minor") == 1
    assert severity_label_to_index("major") == 2
    assert severity_label_to_index("critical") == 3


def test_unknown_domain():
    """Should raise ValueError for unknown domain."""
    with pytest.raises(ValueError, match="Unknown domain"):
        get_taxonomy("unknown")


def test_unknown_severity_label():
    """Should raise ValueError for unknown severity label."""
    with pytest.raises(ValueError, match="Unknown severity label"):
        severity_label_to_index("unknown")


def test_register_custom_taxonomy():
    """Users should be able to register custom domains."""
    tax = register_taxonomy(
        domain="autonomous_driving",
        cost_levels=[5000, 50000, 500000, 5000000],
        labels=["fender_bender", "injury", "serious_injury", "fatality"],
        description="Autonomous vehicle accident severity",
    )
    assert tax.domain == "autonomous_driving"
    assert len(tax.cost_levels) == 4
    assert tax.labels[3] == "fatality"

    # Should be retrievable
    retrieved = get_taxonomy("autonomous_driving")
    assert retrieved.domain == "autonomous_driving"

    # Should appear in list_domains
    assert "autonomous_driving" in list_domains()


def test_register_taxonomy_auto_labels():
    """Custom taxonomy with 3 levels should auto-generate labels."""
    tax = register_taxonomy(
        domain="custom_3level",
        cost_levels=[10, 100, 1000],
    )
    assert tax.labels == ["level_1", "level_2", "level_3"]
    assert len(tax.cost_levels) == 3


def test_register_taxonomy_mismatched_labels():
    """Should reject mismatched labels/cost_levels."""
    with pytest.raises(ValueError, match="len\\(labels\\)"):
        register_taxonomy(
            domain="bad",
            cost_levels=[100, 1000],
            labels=["a", "b", "c"],
        )


def test_list_domains_includes_builtins():
    """list_domains() should include the 5 built-in domains."""
    domains = list_domains()
    for d in ["finance", "medical", "legal", "code_security", "moderation"]:
        assert d in domains
