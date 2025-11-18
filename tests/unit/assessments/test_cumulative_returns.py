"""Tests for Cumulative Returns assessment."""

import numpy as np
import pandas as pd

from src.assessments.cumulative_returns import CumulativeReturns


def test_cumulative_returns_positive():
    """Test cumulative returns with positive returns."""
    # Returns: 10%, 10%, 10%
    # Cumulative: (1.10 * 1.10 * 1.10) - 1 = 1.331 - 1 = 0.331 (33.1%)
    returns = pd.Series([0.10, 0.10, 0.10])

    result = CumulativeReturns._summary(returns=returns)

    expected = 1.10**3 - 1
    assert np.isclose(result, expected, rtol=0.01)


def test_cumulative_returns_negative():
    """Test cumulative returns with negative returns."""
    # Returns: -10%, -10%, -10%
    # Cumulative: (0.90 * 0.90 * 0.90) - 1 = 0.729 - 1 = -0.271 (-27.1%)
    returns = pd.Series([-0.10, -0.10, -0.10])

    result = CumulativeReturns._summary(returns=returns)

    expected = 0.90**3 - 1
    assert np.isclose(result, expected, rtol=0.01)


def test_cumulative_returns_zero():
    """Test cumulative returns with zero returns."""
    returns = pd.Series([0.0, 0.0, 0.0])

    result = CumulativeReturns._summary(returns=returns)

    assert result == 0.0


def test_cumulative_returns_mixed():
    """Test cumulative returns with mixed returns."""
    # Returns: 20%, -10%, 5%
    # Cumulative: (1.20 * 0.90 * 1.05) - 1 = 1.134 - 1 = 0.134 (13.4%)
    returns = pd.Series([0.20, -0.10, 0.05])

    result = CumulativeReturns._summary(returns=returns)

    expected = (1.20 * 0.90 * 1.05) - 1
    assert np.isclose(result, expected, rtol=0.01)


def test_cumulative_returns_round_trip():
    """Test cumulative returns with round trip (up then down)."""
    # Returns: 50%, -33.33% = back to starting point
    # Cumulative: (1.50 * 0.6667) - 1 ≈ 0
    returns = pd.Series([0.50, -1 / 3])

    result = CumulativeReturns._summary(returns=returns)

    # Should be close to 0
    assert np.isclose(result, 0.0, atol=0.01)
