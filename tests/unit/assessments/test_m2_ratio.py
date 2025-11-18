"""Tests for M² (Modigliani-Modigliani) Ratio assessment."""

import numpy as np
import pandas as pd

from src.assessments.m2_ratio import M2Ratio


def test_m2_ratio_outperformance():
    """Test M² ratio with portfolio outperforming benchmark."""
    # Portfolio: higher returns, similar volatility
    # Returns: 4%, 5%, 6% (mean = 5%, annualized = 5% * 252 = 1260%)
    # Benchmark: 2%, 3%, 4% (mean = 3%, annualized = 3% * 252 = 756%)
    # RFR: 1%, 1%, 1%
    returns = pd.Series([0.04, 0.05, 0.06])
    bmk = pd.Series([0.02, 0.03, 0.04])
    rfr = pd.Series([0.01, 0.01, 0.01])

    result = M2Ratio._summary(returns=returns, bmk=bmk, rfr=rfr, ann_factor=252)

    # M² should be positive (outperformance)
    assert result > 0


def test_m2_ratio_underperformance():
    """Test M² ratio with portfolio underperforming benchmark."""
    # Portfolio: lower returns
    # Returns: 1%, 2%, 3% (mean = 2%)
    # Benchmark: 3%, 4%, 5% (mean = 4%)
    # RFR: 1%, 1%, 1%
    returns = pd.Series([0.01, 0.02, 0.03])
    bmk = pd.Series([0.03, 0.04, 0.05])
    rfr = pd.Series([0.01, 0.01, 0.01])

    result = M2Ratio._summary(returns=returns, bmk=bmk, rfr=rfr, ann_factor=252)

    # M² should be negative (underperformance)
    assert result < 0


def test_m2_ratio_same_performance():
    """Test M² ratio when portfolio matches benchmark."""
    # Same returns as benchmark
    returns = pd.Series([0.02, 0.03, 0.04])
    bmk = pd.Series([0.02, 0.03, 0.04])
    rfr = pd.Series([0.01, 0.01, 0.01])

    result = M2Ratio._summary(returns=returns, bmk=bmk, rfr=rfr, ann_factor=252)

    # M² should be near 0 (same performance)
    assert np.isclose(result, 0, atol=0.1)
