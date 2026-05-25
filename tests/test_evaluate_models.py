"""Tests for the pure-logic helpers in experiments.evaluate_models.

The API client wrappers (`call_*`) and the batch dispatchers are not unit
tested here because they exercise external SDKs; they are smoke-tested via
`experiments/run_local_mixed.sh`. We focus on the deterministic
helpers that drive scoring and per-model configuration.
"""

from __future__ import annotations

import pytest

from experiments.evaluate_models import (
    _extract_number,
    _extract_yes_no,
    _is_mcq_match,
    _is_thinking_model,
    _normalize_text,
    _strip_think_tags,
    get_prompt,
    score_prediction,
)

# ----------------------------------------------------------------------
# _extract_number — handles dollar amounts, percentages, raw numbers
# ----------------------------------------------------------------------


def test_extract_number_dollar_billion():
    assert _extract_number("$14.2 billion") == 14_200.0  # scaled to millions


def test_extract_number_dollar_trillion():
    assert _extract_number("$2 trillion") == 2_000_000.0


def test_extract_number_dollar_million_default():
    # Plain dollar amount with no scale is kept as-is (most datasets are in M)
    assert _extract_number("$1577") == 1577.0


def test_extract_number_with_commas():
    assert _extract_number("$1,577.5 million") == 1577.5


def test_extract_number_percentage():
    assert _extract_number("32%") == 32.0


def test_extract_number_negative_percentage():
    assert _extract_number("-1.7%") == -1.7


def test_extract_number_skips_year():
    """2018 should be skipped — only the second number is returned."""
    assert _extract_number("In 2018, the value was 42.5") == 42.5


def test_extract_number_year_alone_is_skipped():
    """A standalone year produces no number (interpretable as no figure)."""
    assert _extract_number("Reported in 2018") is None


def test_extract_number_year_with_decimal_returned():
    """A year-like number with a decimal point is treated as a real value."""
    assert _extract_number("Value 2018.5") == 2018.5


def test_extract_number_empty_text():
    assert _extract_number("") is None


def test_extract_number_none_text():
    assert _extract_number(None) is None


def test_extract_number_no_digits():
    assert _extract_number("hello world") is None


def test_extract_number_negative_int():
    assert _extract_number("-42") == -42.0


# ----------------------------------------------------------------------
# _normalize_text
# ----------------------------------------------------------------------


def test_normalize_text_lowercases():
    assert _normalize_text("Hello WORLD") == "hello world"


def test_normalize_text_strips_punctuation():
    assert _normalize_text("Yes!  No?") == "yes no"


def test_normalize_text_collapses_spaces():
    assert _normalize_text("  a    b   c  ") == "a b c"


def test_normalize_text_empty():
    assert _normalize_text("") == ""


# ----------------------------------------------------------------------
# _extract_yes_no
# ----------------------------------------------------------------------


def test_extract_yes_no_yes():
    assert _extract_yes_no("Yes, the company is profitable.") == "yes"


def test_extract_yes_no_no():
    assert _extract_yes_no("No, it is not.") == "no"


def test_extract_yes_no_case_insensitive():
    assert _extract_yes_no("YES") == "yes"


def test_extract_yes_no_avoids_false_positive_not():
    """'Not applicable' should not be detected as a 'no'."""
    assert _extract_yes_no("Not applicable") is None


def test_extract_yes_no_avoids_none_of_the_above():
    """'None of the above' shouldn't trigger a 'no'."""
    assert _extract_yes_no("None of the above") is None


def test_extract_yes_no_not_at_start():
    """A 'yes/no' not at the start is ignored."""
    assert _extract_yes_no("Maybe yes") is None


def test_extract_yes_no_empty():
    assert _extract_yes_no("") is None


# ----------------------------------------------------------------------
# _is_mcq_match
# ----------------------------------------------------------------------


def test_mcq_match_letter():
    assert _is_mcq_match("A", "A. Some text")


def test_mcq_match_letter_lowercase():
    assert _is_mcq_match("a", "A")


def test_mcq_match_text_substring():
    assert _is_mcq_match("The answer is the right option", "the right option")


def test_mcq_match_different_letter_rejected():
    assert not _is_mcq_match("A", "B. Other option")


# ----------------------------------------------------------------------
# score_prediction — the cascade
# ----------------------------------------------------------------------


def test_score_empty_prediction():
    out = score_prediction("", "answer")
    assert out["correct"] is False
    assert out["score_method"] == "empty"


def test_score_empty_reference():
    out = score_prediction("answer", "")
    assert out["correct"] is False
    assert out["score_method"] == "empty"


def test_score_mcq_correct():
    out = score_prediction("A", "A", options={"A": "x", "B": "y"})
    assert out["correct"] is True
    assert out["score_method"] == "mcq"


def test_score_mcq_wrong():
    out = score_prediction("B", "A", options={"A": "x", "B": "y"})
    assert out["correct"] is False
    assert out["score_method"] == "mcq"


def test_score_yes_no_correct():
    out = score_prediction("Yes, indeed.", "yes, the company")
    assert out["correct"] is True
    assert out["score_method"] == "yes_no"


def test_score_yes_no_wrong():
    out = score_prediction("No, it is not.", "yes")
    assert out["correct"] is False
    assert out["score_method"] == "yes_no"


def test_score_numeric_within_tolerance():
    """A 5% relative error counts as correct."""
    out = score_prediction("$1,500", "$1,549")
    assert out["correct"] is True
    assert out["score_method"] == "numeric"


def test_score_numeric_outside_tolerance():
    """A 10% relative error counts as wrong."""
    out = score_prediction("$1,400", "$1,549")
    assert out["correct"] is False
    assert out["score_method"] == "numeric"


def test_score_numeric_zero_reference():
    """Reference 0 with non-zero prediction is wrong."""
    out = score_prediction("0.5", "0")
    assert out["correct"] is False
    assert out["score_method"] == "numeric"


def test_score_numeric_zero_match():
    """Both 0 → correct."""
    out = score_prediction("0", "0")
    assert out["correct"] is True


def test_score_exact_text_match():
    out = score_prediction("hello world", "Hello, World!")
    assert out["correct"] is True
    assert out["score_method"] == "exact"


def test_score_fuzzy_contains():
    out = score_prediction(
        "The full clause states that liability is limited", "liability is limited"
    )
    assert out["correct"] is True
    assert out["score_method"] == "fuzzy_contains"


def test_score_fuzzy_words():
    out = score_prediction(
        "a sentence with many words about liability and contracts",
        "many liability contracts",
    )
    # 3 words required minimum — 'many', 'liability', 'contracts' all present
    assert out["correct"] is True
    assert out["score_method"] == "fuzzy_words"


def test_score_no_match():
    out = score_prediction("completely unrelated text", "expected answer")
    assert out["correct"] is False
    assert out["score_method"] == "no_match"


def test_score_short_reference_no_fuzzy():
    """Reference shorter than 4 chars must not trigger fuzzy_contains."""
    out = score_prediction("this contains no answer", "no")
    # 'no' matches via yes_no path before fuzzy.
    assert out["score_method"] == "yes_no"


def test_score_short_reference_no_fuzzy_when_not_yes_no():
    """Non-yes-no short reference shouldn't false-positive on substring."""
    out = score_prediction("there is a cat", "ax")
    assert out["correct"] is False


# ----------------------------------------------------------------------
# _is_thinking_model and _strip_think_tags
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_id, expected",
    [
        ("unsloth/QwQ-32B-unsloth-bnb-4bit", True),
        ("Qwen/Qwen3-30B-A3B", True),
        ("qwen/qwen3-235b-a22b-thinking-2507", True),
        ("unsloth/DeepSeek-R1-Distill-Llama-70B-bnb-4bit", True),
        ("unsloth/Llama-3.3-70B-Instruct-bnb-4bit", False),
        ("unsloth/phi-4-bnb-4bit", False),
        ("o3-2025-04-16", False),
    ],
)
def test_is_thinking_model(model_id: str, expected: bool):
    assert _is_thinking_model(model_id) == expected


def test_strip_think_tags_present():
    raw = "<think>I should compute X then Y</think>The answer is 42."
    assert _strip_think_tags(raw) == "The answer is 42."


def test_strip_think_tags_missing():
    """No closing tag → return as-is (truncation case)."""
    raw = "<think>I'm reasoning but got truncated"
    assert _strip_think_tags(raw) == raw


def test_strip_think_tags_no_tag():
    """No <think> at all → return as-is."""
    raw = "Just an answer"
    assert _strip_think_tags(raw) == raw


def test_strip_think_tags_multiline():
    raw = "<think>line1\nline2\nline3</think>\n\nFinal: 7"
    assert _strip_think_tags(raw) == "Final: 7"


# ----------------------------------------------------------------------
# get_prompt — template substitution
# ----------------------------------------------------------------------


def test_get_prompt_known_dataset_original():
    out = get_prompt(
        "What is X?", "financebench", evidence="Some filing", prompt_style="original"
    )
    assert "What is X?" in out
    assert "Some filing" in out
    assert "[START OF FILING]" in out


def test_get_prompt_unknown_dataset_uses_default():
    out = get_prompt("Question?", "nonexistent_dataset", evidence="ctx")
    assert "Question?" in out


def test_get_prompt_mcq_options_rendered():
    out = get_prompt(
        "Pick one",
        "medqa",
        options={"A": "first", "B": "second"},
    )
    assert "A. first" in out
    assert "B. second" in out


def test_get_prompt_handles_braces_in_evidence():
    """Evidence containing stray curly braces (common in legal text) must not crash."""
    out = get_prompt("Q?", "cuad", evidence="The clause says {whatever} and continues.")
    assert "{whatever}" in out


def test_get_prompt_standard_style():
    out_original = get_prompt(
        "Q?", "financebench", evidence="ev", prompt_style="original"
    )
    out_standard = get_prompt(
        "Q?", "financebench", evidence="ev", prompt_style="standard"
    )
    # Different templates produce different output
    assert out_original != out_standard


def test_get_prompt_no_evidence_no_filing_block():
    """Without evidence the {context} block is empty and not wrapped."""
    out = get_prompt("Q?", "medqa", options={"A": "x", "B": "y"})
    assert "Document excerpt" not in out
