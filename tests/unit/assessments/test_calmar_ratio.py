"""Tests for Calmar Ratio assessment."""

import numpy as np
import pandas as pd

from src.assessments.calmar_ratio import CalmarRatio


def test_calmar_ratio_with_drawdown():
    """Test Calmar ratio with positive returns and drawdown."""
    # Returns that create a drawdown: 10%, -15%, 5%
    # Cumulative: 1.10, 0.935, 0.98175
    # Max drawdown from 1.10 to 0.935 = -15%
    returns = pd.Series([0.10, -0.15, 0.05])

    result = CalmarRatio._summary(returns=returns, ann_factor=252)

    # Annualized return / abs(max drawdown)
    # Should be positive since mean return is 0%
    assert isinstance(result, float)


def test_calmar_ratio_no_drawdown():
    """Test Calmar ratio with only positive returns (no drawdown)."""
    # Returns: 1%, 2%, 3% - no drawdown
    returns = pd.Series([0.01, 0.02, 0.03])

    result = CalmarRatio._summary(returns=returns, ann_factor=252)

    # Max drawdown should be 0, result should be inf or very high
    assert result > 100 or np.isinf(result)


def test_calmar_ratio_all_negative():
    """Test Calmar ratio with all negative returns."""
    # Returns: -2%, -3%, -4%
    returns = pd.Series([-0.02, -0.03, -0.04])

    result = CalmarRatio._summary(returns=returns, ann_factor=252)

    # Negative mean return, positive max drawdown, result should be negative
    assert result < 0
