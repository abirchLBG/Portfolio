"""Tests for Sortino Ratio assessment."""

import pandas as pd

from src.assessments.sortino_ratio import SortinoRatio


def test_sortino_ratio_with_downside():
    """Test Sortino ratio with both positive and negative returns."""
    # Returns: -2%, 0%, 4% (mean = 0.67%)
    # Target: 0%
    # Downside deviations: -2%, 0, 0
    # Downside variance of deviations from target
    returns = pd.Series([-0.02, 0.00, 0.04])
    target = 0.0

    result = SortinoRatio._summary(returns=returns, target=target, ann_factor=252)

    # Should be positive since mean return > target
    assert result > 0


def test_sortino_ratio_all_positive_returns():
    """Test Sortino ratio when all returns are above target."""
    # Returns: 2%, 3%, 4% (mean = 3%)
    # Target: 1%
    # All returns above target, no downside deviation
    returns = pd.Series([0.02, 0.03, 0.04])
    target = 0.01

    result = SortinoRatio._summary(returns=returns, target=target, ann_factor=252)

    # Should be very high or inf since no downside risk
    assert result > 0


def test_sortino_ratio_all_negative_returns():
    """Test Sortino ratio when all returns are below target."""
    # Returns: -3%, -2%, -1% (mean = -2%)
    # Target: 0%
    # All returns below target
    returns = pd.Series([-0.03, -0.02, -0.01])
    target = 0.0

    result = SortinoRatio._summary(returns=returns, target=target, ann_factor=252)

    # Should be negative since mean return < target
    assert result < 0
