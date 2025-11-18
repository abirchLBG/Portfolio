"""Tests for CAGR assessment."""

import numpy as np
import pandas as pd

from src.assessments.cagr import CAGR


def test_cagr_positive_growth():
    """Test CAGR with positive growth."""
    # Returns: 10%, 10%, 10%
    # Cumulative: 1.10 * 1.10 * 1.10 = 1.331
    # CAGR = (1.331)^(1/3) - 1 = 0.10 (10%)
    returns = pd.Series([0.10, 0.10, 0.10])

    result = CAGR._summary(returns=returns, ann_factor=1)

    # CAGR should be close to 10% when annualization is 1
    assert np.isclose(result, 0.10, rtol=0.01)


def test_cagr_zero_returns():
    """Test CAGR with zero returns."""
    # No growth
    returns = pd.Series([0.0, 0.0, 0.0])

    result = CAGR._summary(returns=returns, ann_factor=1)

    # CAGR should be 0
    assert np.isclose(result, 0.0, atol=0.001)


def test_cagr_negative_returns():
    """Test CAGR with negative returns."""
    # Returns: -10%, -10%, -10%
    # Cumulative: 0.90 * 0.90 * 0.90 = 0.729
    # CAGR = (0.729)^(1/3) - 1 ≈ -0.10
    returns = pd.Series([-0.10, -0.10, -0.10])

    result = CAGR._summary(returns=returns, ann_factor=1)

    # CAGR should be close to -10%
    assert np.isclose(result, -0.10, rtol=0.01)


def test_cagr_mixed_returns():
    """Test CAGR with mixed returns."""
    # Returns: 20%, -10%, 5%
    # Cumulative: 1.20 * 0.90 * 1.05 = 1.134
    # CAGR = (1.134)^(1/3) - 1 ≈ 0.0429 (4.29%)
    returns = pd.Series([0.20, -0.10, 0.05])

    result = CAGR._summary(returns=returns, ann_factor=1)

    # CAGR should be positive but less than arithmetic mean
    assert result > 0
    assert result < 0.05  # Less than simple average
