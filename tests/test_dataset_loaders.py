"""Tests for the new severity loaders: MedMCQA, DDI Corpus, PrivacyQA.

The full ``load_*()`` functions download from HuggingFace and are
exercised end-to-end by the smoke runs. These tests cover the pure
deterministic helpers that decide which severity tier each instance
receives, because that mapping directly drives the per-domain
aggregates (Kendall tau, variance decomposition, routing) in the paper.
A regression in the severity-mapping logic would silently rotate the
hypothesis-test numbers.
"""

from __future__ import annotations

import pandas as pd
import pytest

from experiments.datasets.load_ddi import (
    RELATION_SEVERITY,
    _normalise_relation,
    classify_severity as ddi_severity,
)
from experiments.datasets.load_medmcqa import (
    SUBJECT_SEVERITY,
    classify_severity as medmcqa_severity,
)
from experiments.datasets.load_privacyqa import (
    CATEGORY_SEVERITY,
    _coerce_columns,
    classify_question_category,
    classify_severity as privacyqa_severity,
)


VALID_SEVERITIES = {"negligible", "minor", "major", "critical"}


# ----------------------------------------------------------------------
# MedMCQA: subject -> severity
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "subject,expected",
    [
        ("Pharmacology", "critical"),
        ("Surgery", "critical"),
        ("Anaesthesia", "critical"),
        ("Pathology", "major"),
        ("Microbiology", "major"),
        ("Psychiatry", "minor"),
        ("Forensic Medicine", "minor"),
        ("Anatomy", "negligible"),
        ("Biochemistry", "negligible"),
        # Case-insensitivity
        ("pharmacology", "critical"),
        ("ANATOMY", "negligible"),
    ],
)
def test_medmcqa_severity_known_subjects(subject, expected):
    assert medmcqa_severity(subject) == expected


def test_medmcqa_severity_unknown_subject_defaults_to_minor():
    """An unseen subject must NOT silently get critical or negligible."""
    assert medmcqa_severity("Cardiology") == "minor"
    assert medmcqa_severity("") == "minor"


def test_medmcqa_severity_all_subjects_map_to_valid_tier():
    """Every value in the subject table must be a recognised tier."""
    invalid = {s for s in SUBJECT_SEVERITY.values() if s not in VALID_SEVERITIES}
    assert invalid == set(), f"Invalid severity tiers in MedMCQA map: {invalid}"


def test_medmcqa_severity_covers_critical_subjects():
    """Direct-iatrogenic-harm subjects must be tier-critical: changing one
    of these to minor would silently deflate the medical domain's variance.
    """
    must_be_critical = {
        "pharmacology",
        "anaesthesia",
        "surgery",
        "medicine",
        "pediatrics",
    }
    for s in must_be_critical:
        assert SUBJECT_SEVERITY[s] == "critical", (
            f"{s} must be critical (cost-of-error baseline)"
        )


# ----------------------------------------------------------------------
# DDI Corpus: relation -> severity
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "relation,expected_norm",
    [
        ("effect", "effect"),
        ("DDI-effect", "effect"),
        ("Advice", "advice"),
        ("DDI-advise", "advice"),
        ("mechanism", "mechanism"),
        ("int", "int"),
        ("false", "none"),
        ("None", "none"),
        ("", "none"),
        ("unknown_label", "none"),
    ],
)
def test_ddi_normalise_relation_handles_dataset_variants(relation, expected_norm):
    assert _normalise_relation(relation) == expected_norm


def test_ddi_severity_critical_only_on_effect():
    """Only ``effect`` should carry critical severity: it's the relation
    whose miss has direct clinical impact."""
    critical = {
        r for r, tier in RELATION_SEVERITY.items() if tier == "critical"
    }
    # Accept both canonical and DDI-prefixed variants.
    assert critical == {"effect", "ddi-effect"}


def test_ddi_severity_negligible_only_on_no_interaction():
    """A spurious alert on a non-interacting pair is operationally cheap;
    nothing else should land in negligible."""
    neg = {r for r, tier in RELATION_SEVERITY.items() if tier == "negligible"}
    assert neg == {"false", "ddi-false", "none"}


def test_ddi_severity_dispatch_matches_normalised_label():
    assert ddi_severity("DDI-effect") == "critical"
    assert ddi_severity("Advice") == "major"
    assert ddi_severity("False") == "negligible"
    assert ddi_severity("") == "negligible"  # absent -> no interaction


def test_ddi_severity_all_tiers_valid():
    invalid = {s for s in RELATION_SEVERITY.values() if s not in VALID_SEVERITIES}
    assert invalid == set()


# ----------------------------------------------------------------------
# PrivacyQA: question -> OPP-115 category -> severity
# ----------------------------------------------------------------------


def test_privacyqa_data_security_question_routes_to_critical():
    """Data-security questions must map to critical (GDPR Art.32 tier).

    This is the only PrivacyQA tier that triggers the highest cost level;
    misrouting these inflates/deflates the insurance domain's E[S]."""
    qs = [
        "How is my password stored?",
        "Is my data encrypted in transit?",
        "Has the company disclosed any security breach?",
    ]
    for q in qs:
        sev, cat = privacyqa_severity(q)
        assert cat == "Data Security", f"{q!r} -> {cat}"
        assert sev == "critical"


def test_privacyqa_third_party_question_routes_to_major():
    qs = [
        "Does the app share my data with third parties?",
        "Are my personal details sold to advertisers?",
        "Which partners receive my information?",
    ]
    for q in qs:
        sev, cat = privacyqa_severity(q)
        assert cat == "Third Party Sharing/Collection", f"{q!r} -> {cat}"
        assert sev == "major"


def test_privacyqa_unmatched_question_falls_back_to_other_negligible():
    """A question that fires no OPP-115 rule must land in Other/negligible
    rather than being silently mapped to a high tier."""
    sev, cat = privacyqa_severity("xyz nonsense placeholder")
    assert cat == "Other"
    assert sev == "negligible"


def test_privacyqa_keyword_specificity_security_beats_collection():
    """The phrase 'collect my password securely' must route to Data
    Security, not First Party Collection, because security rules are
    checked first per the OPP-115 source ordering."""
    sev, cat = privacyqa_severity("Does the app collect my password securely?")
    assert cat == "Data Security"
    assert sev == "critical"


def test_privacyqa_severity_all_categories_valid_tier():
    invalid = {s for s in CATEGORY_SEVERITY.values() if s not in VALID_SEVERITIES}
    assert invalid == set()


def test_privacyqa_classify_returns_known_category():
    """The fallback path should always return one of the 10 OPP-115
    categories (or Other), never an arbitrary string."""
    seen_cats = set()
    for q in [
        "How is my password stored?",
        "How long do you keep my data?",
        "Can I delete my account?",
        "I want to opt out of marketing.",
        "Tell me about your data sharing policy.",
        "When was this policy updated?",
        "Do you track me across the web?",
        "Are children allowed to use the app?",
        "Garbage input.",
    ]:
        cat = classify_question_category(q)
        seen_cats.add(cat)
        assert cat in CATEGORY_SEVERITY, cat
    assert seen_cats != {"Other"}, "Keyword rules never fired -- check coverage"


def test_privacyqa_coerce_columns_accepts_official_csv_schema():
    """The official CSV headers must be normalised to the loader's
    expected ``query / segment / label`` triple. A regression here would
    silently drop the entire dataset to zero rows."""
    raw = pd.DataFrame(
        {
            "QueryID": ["q1"],
            "DocID": ["app1"],
            "Query": ["Does the app track my location?"],
            "SegmentID": ["s5"],
            "Segment": ["We collect approximate location for analytics."],
            "Any_Relevant": ["Y"],
        }
    )
    out = _coerce_columns(raw)
    assert set(["query", "segment", "label"]).issubset(out.columns)
    assert out.loc[0, "label"] == "Y"


def test_privacyqa_coerce_columns_normalises_label_values():
    """``Yes``, ``1``, ``True``, ``relevant`` must all become Y; any other
    value (No, n, false, blank) must become N."""
    raw = pd.DataFrame(
        {
            "query": ["q"] * 6,
            "segment": ["s"] * 6,
            "label": ["yes", "1", "True", "relevant", "no", ""],
        }
    )
    out = _coerce_columns(raw)
    assert out["label"].tolist() == ["Y", "Y", "Y", "Y", "N", "N"]


def test_privacyqa_coerce_columns_raises_when_required_missing():
    """If the input frame is missing one of the required columns, the
    loader must FAIL LOUDLY rather than silently produce empty data."""
    raw = pd.DataFrame({"query": ["q"], "label": ["Y"]})  # segment missing
    with pytest.raises(ValueError, match="segment"):
        _coerce_columns(raw)


# ----------------------------------------------------------------------
# Cross-loader invariants
# ----------------------------------------------------------------------


def test_all_loaders_use_canonical_severity_labels():
    """The three new loaders must use exactly the 4-tier label space the
    rest of the framework expects (severity_label_to_index relies on it)."""
    for table in (SUBJECT_SEVERITY, RELATION_SEVERITY, CATEGORY_SEVERITY):
        for tier in table.values():
            assert tier in VALID_SEVERITIES, tier
