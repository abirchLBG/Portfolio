"""Tests for Tracking Error assessment."""

import numpy as np
import pandas as pd

from src.assessments.tracking_error import TrackingError


def test_tracking_error_perfect_tracking():
    """Test tracking error when portfolio perfectly tracks benchmark."""
    # Identical returns
    returns = pd.Series([0.01, 0.02, 0.03])
    bmk = pd.Series([0.01, 0.02, 0.03])

    result = TrackingError._summary(returns=returns, bmk=bmk, ann_factor=252)

    # Tracking error should be 0
    assert result == 0.0


def test_tracking_error_constant_difference():
    """Test tracking error with constant outperformance."""
    # Returns consistently 1% higher
    # Active returns: 1%, 1%, 1%
    # Std dev of active returns = 0
    returns = pd.Series([0.02, 0.03, 0.04])
    bmk = pd.Series([0.01, 0.02, 0.03])

    result = TrackingError._summary(returns=returns, bmk=bmk, ann_factor=252)

    # Constant difference, no variation, tracking error = 0 (with floating point tolerance)
    assert np.isclose(result, 0.0, atol=1e-10)


def test_tracking_error_variable_difference():
    """Test tracking error with variable tracking difference."""
    # Returns: 0%, 2%, 4%
    # Benchmark: 1%, 2%, 3%
    # Active: -1%, 0%, 1%
    # Std dev of active > 0
    returns = pd.Series([0.00, 0.02, 0.04])
    bmk = pd.Series([0.01, 0.02, 0.03])

    result = TrackingError._summary(returns=returns, bmk=bmk, ann_factor=252)

    # Should be positive
    assert result > 0
