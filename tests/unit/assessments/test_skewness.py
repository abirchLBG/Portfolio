"""Tests for Skewness assessment."""

import numpy as np
import pandas as pd

from src.assessments.skewness import Skewness


def test_skewness_symmetric():
    """Test skewness with symmetric distribution."""
    # Symmetric returns: -2%, -1%, 0%, 1%, 2%
    # Skewness should be near 0
    returns = pd.Series([-0.02, -0.01, 0.00, 0.01, 0.02])

    result = Skewness._summary(returns=returns)

    # Should be close to 0 for symmetric distribution
    assert np.isclose(result, 0, atol=0.3)


def test_skewness_right_skewed():
    """Test skewness with right-skewed distribution (positive skew)."""
    # Most returns low, one high outlier: 1%, 1%, 1%, 1%, 10%
    returns = pd.Series([0.01, 0.01, 0.01, 0.01, 0.10])

    result = Skewness._summary(returns=returns)

    # Should be positive (right tail is longer)
    assert result > 0


def test_skewness_left_skewed():
    """Test skewness with left-skewed distribution (negative skew)."""
    # Most returns high, one low outlier: -10%, 1%, 1%, 1%, 1%
    returns = pd.Series([-0.10, 0.01, 0.01, 0.01, 0.01])

    result = Skewness._summary(returns=returns)

    # Should be negative (left tail is longer)
    assert result < 0


def test_skewness_constant():
    """Test skewness with constant returns."""
    # All same value
    returns = pd.Series([0.01, 0.01, 0.01])

    result = Skewness._summary(returns=returns)

    # Skewness undefined for constant values, should be NaN or 0
    assert np.isnan(result) or result == 0
