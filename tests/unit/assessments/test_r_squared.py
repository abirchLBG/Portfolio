"""Tests for R-Squared assessment."""

import numpy as np
import pandas as pd

from src.assessments.r_squared import RSquared


def test_r_squared_perfect_correlation():
    """Test R² with perfect positive correlation."""
    # Returns: 2%, 4%, 8%
    # Benchmark: 1%, 2%, 4%
    # Perfect linear relationship, correlation = 1, R² = 1
    returns = pd.Series([0.02, 0.04, 0.08])
    bmk = pd.Series([0.01, 0.02, 0.04])

    result = RSquared._summary(returns=returns, bmk=bmk)

    # R² should be 1.0
    assert np.isclose(result, 1.0, rtol=0.01)


def test_r_squared_no_correlation():
    """Test R² with no correlation."""
    # Returns constant, benchmark varies
    returns = pd.Series([0.02, 0.02, 0.02])
    bmk = pd.Series([0.01, 0.02, 0.03])

    result = RSquared._summary(returns=returns, bmk=bmk)

    # R² should be 0 or NaN (no variation in returns)
    assert np.isnan(result) or np.isclose(result, 0, atol=0.1)


def test_r_squared_partial_correlation():
    """Test R² with partial correlation."""
    # Some relationship but not perfect
    returns = pd.Series([0.01, 0.02, 0.03, 0.05])
    bmk = pd.Series([0.01, 0.02, 0.04, 0.03])

    result = RSquared._summary(returns=returns, bmk=bmk)

    # Should be between 0 and 1
    assert 0 <= result <= 1


def test_r_squared_negative_correlation():
    """Test R² with negative correlation."""
    # Perfect negative correlation: correlation = -1, R² = 1
    # Perfect negative: returns = -1 * bmk + 0.05
    returns = pd.Series([0.04, 0.03, 0.01])
    bmk = pd.Series([0.01, 0.02, 0.04])

    result = RSquared._summary(returns=returns, bmk=bmk)

    # R² should still be 1.0 (measures strength, not direction)
    assert np.isclose(result, 1.0, rtol=0.01)
