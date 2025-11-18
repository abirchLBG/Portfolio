"""Tests for Semi-Variance assessment."""

import pandas as pd

from src.assessments.semi_variance import SemiVariance


def test_semi_variance_all_positive():
    """Test semi-variance when all returns are above target."""
    # Returns: 2%, 3%, 4% all above 0% target
    # No downside deviation
    returns = pd.Series([0.02, 0.03, 0.04])

    result = SemiVariance._summary(returns=returns, target=0.0, ann_factor=252)

    # Semi-variance should be 0 (no returns below target)
    assert result == 0.0


def test_semi_variance_all_negative():
    """Test semi-variance when all returns are below target."""
    # Returns: -3%, -2%, -1% all below 0% target
    # All contribute to downside deviation
    returns = pd.Series([-0.03, -0.02, -0.01])

    result = SemiVariance._summary(returns=returns, target=0.0, ann_factor=252)

    # Semi-variance should be positive and annualized
    assert result > 0


def test_semi_variance_mixed_returns():
    """Test semi-variance with mixed returns."""
    # Returns: -2%, 0%, 2% with target = 0%
    # Only -2% contributes to downside
    returns = pd.Series([-0.02, 0.00, 0.02])

    result = SemiVariance._summary(returns=returns, target=0.0, ann_factor=252)

    # Should be positive but less than total variance
    assert result > 0
    total_variance = returns.var() * 252
    assert result < total_variance


def test_semi_variance_custom_target():
    """Test semi-variance with custom target."""
    # Returns: 1%, 2%, 3% with target = 2%
    # Only 1% is below target
    returns = pd.Series([0.01, 0.02, 0.03])
    target = 0.02

    result = SemiVariance._summary(returns=returns, target=target, ann_factor=252)

    # Should be positive (1% return is below 2% target)
    assert result > 0
