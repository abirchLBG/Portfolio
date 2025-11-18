"""Tests for Treynor Ratio assessment."""

import numpy as np
import pandas as pd

from src.assessments.treynor_ratio import TreynorRatio


def test_treynor_ratio_positive_beta():
    """Test Treynor ratio with positive beta."""
    # Returns: 4%, 8% (mean = 6%)
    # Benchmark: 2%, 4% (mean = 3%)
    # RFR: 1%, 1%
    # Beta ≈ 2 (returns move 2x benchmark)
    # Excess return: 5%, Treynor = 5% / 2 = 2.5%
    returns = pd.Series([0.04, 0.08])
    bmk = pd.Series([0.02, 0.04])
    rfr = pd.Series([0.01, 0.01])

    result = TreynorRatio._summary(returns=returns, bmk=bmk, rfr=rfr, ann_factor=252)

    # Should be positive
    assert result > 0


def test_treynor_ratio_negative_excess():
    """Test Treynor ratio with negative excess returns."""
    # Returns below RFR
    returns = pd.Series([0.00, 0.005])
    bmk = pd.Series([0.01, 0.02])
    rfr = pd.Series([0.01, 0.01])

    result = TreynorRatio._summary(returns=returns, bmk=bmk, rfr=rfr, ann_factor=252)

    # Negative excess return should give negative Treynor
    assert result < 0


def test_treynor_ratio_zero_beta():
    """Test Treynor ratio with zero beta (no correlation)."""
    # Returns constant, benchmark varies
    returns = pd.Series([0.02, 0.02, 0.02])
    bmk = pd.Series([0.01, 0.02, 0.03])
    rfr = pd.Series([0.01, 0.01, 0.01])

    result = TreynorRatio._summary(returns=returns, bmk=bmk, rfr=rfr, ann_factor=252)

    # Beta should be near 0, result could be very large or inf
    # Just check it doesn't error
    assert isinstance(result, (float, np.floating))
