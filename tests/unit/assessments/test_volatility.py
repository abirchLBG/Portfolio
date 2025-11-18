"""Tests for Volatility assessment."""

import numpy as np
import pandas as pd

from src.assessments.volatility import Volatility


def test_volatility_constant_returns():
    """Test volatility with constant returns (volatility = 0)."""
    # All returns the same
    returns = pd.Series([0.01, 0.01, 0.01])

    result = Volatility._summary(returns=returns, ann_factor=252)

    # Volatility should be 0
    assert result == 0.0


def test_volatility_simple_case():
    """Test volatility with simple returns."""
    # Returns: -1%, 0%, 1%
    # Std dev = 1%
    # Annualized = 1% * sqrt(252) ≈ 15.87%
    returns = pd.Series([-0.01, 0.00, 0.01])

    result = Volatility._summary(returns=returns, ann_factor=252)

    expected = 0.01 * np.sqrt(252)
    assert np.isclose(result, expected, rtol=0.01)


def test_volatility_higher_variance():
    """Test volatility with higher variance returns."""
    # Returns: -5%, 0%, 5%
    # More volatile
    returns = pd.Series([-0.05, 0.00, 0.05])

    result = Volatility._summary(returns=returns, ann_factor=252)

    # Should be higher than the simple case
    expected = returns.std() * np.sqrt(252)
    assert np.isclose(result, expected, rtol=0.01)
