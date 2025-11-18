"""Tests for VaR (Value at Risk) assessment."""

import numpy as np
import pandas as pd

from src.assessments.var import VaR


def test_var_simple_case():
    """Test VaR with simple returns."""
    # Returns: -3%, -2%, -1%, 0%, 1%, 2%, 3%, 4%, 5%
    # 95% VaR should capture the 5th percentile
    returns = pd.Series([-0.03, -0.02, -0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05])

    result = VaR._summary(returns=returns, confidence_level=0.95)

    # VaR should be negative (it's a loss)
    assert result < 0
    # Should be around -3% (worst 5% of returns)
    assert -0.04 < result < -0.02


def test_var_all_positive():
    """Test VaR with all positive returns."""
    # All positive returns
    returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])

    result = VaR._summary(returns=returns, confidence_level=0.95)

    # VaR could be small positive or negative depending on distribution
    assert isinstance(result, (float, np.floating))


def test_var_different_confidence():
    """Test VaR with different confidence levels."""
    returns = pd.Series([-0.05, -0.03, -0.01, 0.01, 0.03, 0.05])

    var_95 = VaR._summary(returns=returns, confidence_level=0.95)
    var_99 = VaR._summary(returns=returns, confidence_level=0.99)

    # 99% VaR should be more extreme (more negative) than 95% VaR
    assert var_99 <= var_95
