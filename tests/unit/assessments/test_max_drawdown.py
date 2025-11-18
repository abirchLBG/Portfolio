"""Tests for Max Drawdown assessment."""

import pandas as pd

from src.assessments.max_drawdown import MaxDrawdown


def test_max_drawdown_with_decline():
    """Test max drawdown with a clear decline."""
    # Returns: 10%, -20%, 5%
    # Cumulative: 1.10, 0.88, 0.924
    # Drawdown from 1.10 to 0.88 = -20%
    returns = pd.Series([0.10, -0.20, 0.05])

    result = MaxDrawdown._summary(returns=returns)

    # Max drawdown should be negative and around -20%
    assert result < 0
    assert -0.25 < result < -0.15


def test_max_drawdown_no_decline():
    """Test max drawdown with only positive returns."""
    # Returns: 1%, 2%, 3% - always going up
    returns = pd.Series([0.01, 0.02, 0.03])

    result = MaxDrawdown._summary(returns=returns)

    # Max drawdown should be 0 (no peak-to-trough decline)
    assert result == 0.0


def test_max_drawdown_all_negative():
    """Test max drawdown with all negative returns."""
    # Returns: -5%, -5%, -5%
    # Continuous decline from peak of 0.95 to 0.857375
    # Drawdown = (0.857375 - 0.95) / 0.95 = -0.0975
    returns = pd.Series([-0.05, -0.05, -0.05])

    result = MaxDrawdown._summary(returns=returns)

    # Should have significant drawdown around -9.75%
    assert result < -0.09


def test_max_drawdown_recovery():
    """Test max drawdown with recovery."""
    # Returns: 20%, -10%, -5%, 10%
    # Peak at 1.20, trough at ~1.026, recovers to ~1.129
    # Max drawdown from peak
    returns = pd.Series([0.20, -0.10, -0.05, 0.10])

    result = MaxDrawdown._summary(returns=returns)

    # Should capture the peak-to-trough decline
    assert result < 0
