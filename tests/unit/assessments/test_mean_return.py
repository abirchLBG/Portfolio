"""Tests for Mean Return assessment."""

import numpy as np
import pandas as pd

from src.assessments.mean_return import MeanReturn


def test_mean_return_simple():
    """Test mean return with simple case."""
    # Returns: 1%, 2%, 3%
    # Mean = 2%
    # Annualized = 2% * 252 = 504%
    returns = pd.Series([0.01, 0.02, 0.03])

    result = MeanReturn._summary(returns=returns, ann_factor=252)

    expected = 0.02 * 252
    assert np.isclose(result, expected, rtol=0.01)


def test_mean_return_negative():
    """Test mean return with negative returns."""
    # Returns: -1%, -2%, -3%
    # Mean = -2%
    returns = pd.Series([-0.01, -0.02, -0.03])

    result = MeanReturn._summary(returns=returns, ann_factor=252)

    expected = -0.02 * 252
    assert np.isclose(result, expected, rtol=0.01)


def test_mean_return_zero():
    """Test mean return with zero returns."""
    returns = pd.Series([0.0, 0.0, 0.0])

    result = MeanReturn._summary(returns=returns, ann_factor=252)

    assert result == 0.0


def test_mean_return_mixed():
    """Test mean return with mixed returns."""
    # Returns: -2%, 0%, 2%
    # Mean = 0%
    returns = pd.Series([-0.02, 0.00, 0.02])

    result = MeanReturn._summary(returns=returns, ann_factor=252)

    assert np.isclose(result, 0.0, atol=0.001)
