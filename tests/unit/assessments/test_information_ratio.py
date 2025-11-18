"""Tests for Information Ratio assessment."""

import numpy as np
import pandas as pd

from src.assessments.information_ratio import InformationRatio


def test_information_ratio_positive_active_returns():
    """Test Information ratio with positive active returns."""
    # Returns: 3%, 4%, 5% (mean = 4%)
    # Benchmark: 1%, 2%, 3% (mean = 2%)
    # Active: 2%, 2%, 2% (mean = 2%, std = 0)
    returns = pd.Series([0.03, 0.04, 0.05])
    bmk = pd.Series([0.01, 0.02, 0.03])

    result = InformationRatio._summary(returns=returns, bmk=bmk, ann_factor=252)

    # Consistent outperformance, tracking error should be low
    assert result > 0


def test_information_ratio_zero_active_returns():
    """Test Information ratio when returns match benchmark."""
    returns = pd.Series([0.01, 0.02, 0.03])
    bmk = pd.Series([0.01, 0.02, 0.03])

    result = InformationRatio._summary(returns=returns, bmk=bmk, ann_factor=252)

    # Active returns are all zero, should be 0/0 = NaN
    assert np.isnan(result)


def test_information_ratio_negative_active_returns():
    """Test Information ratio with negative active returns (underperformance)."""
    # Returns: 0%, 1%, 2% (mean = 1%)
    # Benchmark: 2%, 3%, 4% (mean = 3%)
    # Active: -2%, -2%, -2% (mean = -2%)
    returns = pd.Series([0.00, 0.01, 0.02])
    bmk = pd.Series([0.02, 0.03, 0.04])

    result = InformationRatio._summary(returns=returns, bmk=bmk, ann_factor=252)

    # Underperformance
    assert result < 0
