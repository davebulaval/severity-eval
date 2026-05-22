"""Tests for severity_eval.validation."""

from __future__ import annotations

import numpy as np
import pytest

from severity_eval.validation import validate_severity_profile


def test_valid_probability_vector_pass_through():
    """A clean probability vector is returned unchanged."""
    pi = [0.4, 0.3, 0.2, 0.1]
    out = validate_severity_profile(pi)
    np.testing.assert_allclose(out, pi)


def test_percentage_auto_converted():
    """A vector summing to ~100 is auto-converted to probabilities."""
    out = validate_severity_profile([40, 30, 20, 10])
    np.testing.assert_allclose(out, [0.4, 0.3, 0.2, 0.1])


def test_negative_values_rejected():
    with pytest.raises(ValueError, match="negative values"):
        validate_severity_profile([0.5, -0.1, 0.6])


def test_nan_rejected():
    """NaN slips past the sum check; explicit guard required."""
    with pytest.raises(ValueError, match="NaN or infinite"):
        validate_severity_profile([0.5, float("nan"), 0.5])


def test_inf_rejected():
    with pytest.raises(ValueError, match="NaN or infinite"):
        validate_severity_profile([0.5, float("inf"), 0.0])


def test_empty_rejected():
    with pytest.raises(ValueError, match="non-empty 1-D"):
        validate_severity_profile([])


def test_two_dim_rejected():
    with pytest.raises(ValueError, match="non-empty 1-D"):
        validate_severity_profile(np.array([[0.5, 0.5]]))


def test_does_not_sum_to_one():
    with pytest.raises(ValueError, match="must sum to 1"):
        validate_severity_profile([0.3, 0.3, 0.2])  # 0.8


def test_zeros_vector_rejected():
    """All-zero vector is rejected (sum != 1, not a probability)."""
    with pytest.raises(ValueError, match="must sum to 1"):
        validate_severity_profile([0.0, 0.0, 0.0])


def test_custom_name_in_error():
    """The 'name' parameter is surfaced in the error message."""
    with pytest.raises(ValueError, match="pi_test"):
        validate_severity_profile([-0.5, 1.5], name="pi_test")


def test_returns_ndarray_not_list():
    """Output is always an ndarray, even when input is a list."""
    out = validate_severity_profile([0.5, 0.5])
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
