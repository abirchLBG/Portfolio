"""Tests for Correlation assessment."""

import numpy as np
import pandas as pd

from src.assessments.correlation import Correlation


def test_correlation_perfect_positive():
    """Test correlation with perfect positive correlation."""
    # Returns: 2%, 4%, 8%
    # Benchmark: 1%, 2%, 4%
    # Perfect linear relationship
    returns = pd.Series([0.02, 0.04, 0.08])
    bmk = pd.Series([0.01, 0.02, 0.04])

    result = Correlation._summary(returns=returns, bmk=bmk)

    # Correlation should be 1.0
    assert np.isclose(result, 1.0, rtol=0.01)


def test_correlation_perfect_negative():
    """Test correlation with perfect negative correlation."""
    # Returns move exactly opposite to benchmark
    # Perfect negative: returns = -1 * bmk + 0.05
    returns = pd.Series([0.04, 0.03, 0.01])
    bmk = pd.Series([0.01, 0.02, 0.04])

    result = Correlation._summary(returns=returns, bmk=bmk)

    # Correlation should be -1.0
    assert np.isclose(result, -1.0, rtol=0.01)


def test_correlation_zero():
    """Test correlation with no linear relationship."""
    # Returns constant, benchmark varies
    returns = pd.Series([0.02, 0.02, 0.02])
    bmk = pd.Series([0.01, 0.02, 0.03])

    result = Correlation._summary(returns=returns, bmk=bmk)

    # Correlation should be 0 or NaN (no variation in returns)
    assert np.isnan(result) or np.isclose(result, 0, atol=0.1)


def test_correlation_partial():
    """Test correlation with partial correlation."""
    # Some positive relationship but not perfect
    returns = pd.Series([0.01, 0.02, 0.04])
    bmk = pd.Series([0.01, 0.03, 0.02])

    result = Correlation._summary(returns=returns, bmk=bmk)

    # Should be between -1 and 1
    assert -1 <= result <= 1
