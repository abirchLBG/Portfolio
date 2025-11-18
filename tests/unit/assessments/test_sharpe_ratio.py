"""Tests for Sharpe Ratio assessment."""

import numpy as np
import pandas as pd

from src.assessments.sharpe_ratio import SharpeRatio


def test_sharpe_ratio_positive_excess_returns():
    """Test Sharpe ratio with positive excess returns."""
    # Returns: 2%, 3%, 4% (mean = 3%)
    # RFR: 1% constant
    # Excess: 1%, 2%, 3% (mean = 2%, std = 1%)
    # Sharpe = 2% * sqrt(252) / 1% = 2 * 15.87 = 31.75
    returns = pd.Series([0.02, 0.03, 0.04])
    rfr = pd.Series([0.01, 0.01, 0.01])

    result = SharpeRatio._summary(returns=returns, rfr=rfr, ann_factor=252)

    # Excess returns: mean = 0.02, std ≈ 0.01
    expected = 0.02 * np.sqrt(252) / 0.01
    assert np.isclose(result, expected, rtol=0.01)


def test_sharpe_ratio_zero_excess_returns():
    """Test Sharpe ratio when returns equal risk-free rate."""
    returns = pd.Series([0.01, 0.01, 0.01])
    rfr = pd.Series([0.01, 0.01, 0.01])

    result = SharpeRatio._summary(returns=returns, rfr=rfr, ann_factor=252)

    # Excess returns are all zero, std is 0, should return NaN
    assert np.isnan(result)


def test_sharpe_ratio_negative_excess_returns():
    """Test Sharpe ratio with negative excess returns."""
    # Returns: -1%, -2%, -3% (mean = -2%)
    # RFR: 1% constant
    # Excess: -2%, -3%, -4% (mean = -3%, std = 1%)
    # Sharpe = -3% * sqrt(252) / 1% = negative
    returns = pd.Series([-0.01, -0.02, -0.03])
    rfr = pd.Series([0.01, 0.01, 0.01])

    result = SharpeRatio._summary(returns=returns, rfr=rfr, ann_factor=252)

    assert result < 0  # Negative Sharpe ratio
