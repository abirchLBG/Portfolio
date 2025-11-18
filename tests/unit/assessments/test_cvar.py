"""Tests for CVaR (Conditional Value at Risk) assessment."""

import numpy as np
import pandas as pd

from src.assessments.cvar import CVaR


def test_cvar_simple_case():
    """Test CVaR with simple returns."""
    # Returns: -5%, -4%, -3%, -2%, -1%, 0%, 1%, 2%, 3%, 4%
    # 95% CVaR is average of worst 5% (just the -5% and -4%)
    returns = pd.Series(
        [-0.05, -0.04, -0.03, -0.02, -0.01, 0.00, 0.01, 0.02, 0.03, 0.04]
    )

    result = CVaR._summary(returns=returns, confidence_level=0.95)

    # CVaR should be more negative than VaR (average of tail)
    assert result < 0


def test_cvar_all_positive():
    """Test CVaR with all positive returns."""
    returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])

    result = CVaR._summary(returns=returns, confidence_level=0.95)

    # CVaR of positive returns
    assert isinstance(result, (float, np.floating))


def test_cvar_vs_var():
    """Test that CVaR is more extreme than VaR."""
    from src.assessments.var import VaR

    returns = pd.Series(
        [-0.10, -0.05, -0.03, -0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05]
    )

    var = VaR._summary(returns=returns, confidence_level=0.95)
    cvar = CVaR._summary(returns=returns, confidence_level=0.95)

    # CVaR should be <= VaR (more extreme loss)
    assert cvar <= var
