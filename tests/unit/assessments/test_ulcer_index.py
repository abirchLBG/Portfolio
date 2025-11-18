"""Tests for Ulcer Index assessment."""

import pandas as pd

from src.assessments.ulcer_index import UlcerIndex


def test_ulcer_index_no_drawdown():
    """Test Ulcer index when there's no drawdown."""
    # Returns: 1%, 2%, 3% - continuously rising
    returns = pd.Series([0.01, 0.02, 0.03])

    result = UlcerIndex._summary(returns=returns)

    # No drawdown, Ulcer index should be 0
    assert result == 0.0


def test_ulcer_index_with_drawdown():
    """Test Ulcer index with drawdown."""
    # Returns: 10%, -15%, -10%, 5%
    # Creates a drawdown period
    returns = pd.Series([0.10, -0.15, -0.10, 0.05])

    result = UlcerIndex._summary(returns=returns)

    # Should be positive (measures drawdown severity)
    assert result > 0


def test_ulcer_index_deep_drawdown():
    """Test Ulcer index with deep drawdown."""
    # Returns: 20%, -30%, -20%
    # Deep drawdown
    returns = pd.Series([0.20, -0.30, -0.20])

    result = UlcerIndex._summary(returns=returns)

    # Should be significantly positive
    assert result > 10  # Ulcer index is in percentage terms


def test_ulcer_index_all_negative():
    """Test Ulcer index with continuous decline."""
    # Returns: -5%, -5%, -5%
    # Continuous drawdown from initial value
    returns = pd.Series([-0.05, -0.05, -0.05])

    result = UlcerIndex._summary(returns=returns)

    # Should be positive and substantial
    assert result > 0
