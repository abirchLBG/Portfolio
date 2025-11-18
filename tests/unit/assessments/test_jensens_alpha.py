"""Tests for Jensen's Alpha assessment."""

import numpy as np
import pandas as pd

from src.assessments.jensens_alpha import JensensAlpha


def test_jensens_alpha_positive():
    """Test Jensen's alpha with outperformance."""
    # Returns: 6%, 10% (mean = 8%)
    # Benchmark: 2%, 4% (mean = 3%)
    # RFR: 1%, 1%
    # Beta ≈ 2
    # Expected return = 1% + 2 * (3% - 1%) = 5%
    # Alpha = 8% - 5% = 3% (positive outperformance)
    returns = pd.Series([0.06, 0.10])
    bmk = pd.Series([0.02, 0.04])
    rfr = pd.Series([0.01, 0.01])

    result = JensensAlpha._summary(returns=returns, bmk=bmk, rfr=rfr, ann_factor=252)

    # Should be positive (outperformance)
    assert result > 0


def test_jensens_alpha_negative():
    """Test Jensen's alpha with underperformance."""
    # Returns: 2%, 4% (mean = 3%)
    # Benchmark: 4%, 8% (mean = 6%)
    # RFR: 1%, 1%
    # Beta ≈ 1
    # Expected return = 1% + 1 * (6% - 1%) = 6%
    # Alpha = 3% - 6% = -3% (underperformance)
    returns = pd.Series([0.02, 0.04])
    bmk = pd.Series([0.04, 0.08])
    rfr = pd.Series([0.01, 0.01])

    result = JensensAlpha._summary(returns=returns, bmk=bmk, rfr=rfr, ann_factor=252)

    # Should be negative (underperformance)
    assert result < 0


def test_jensens_alpha_matches_capm():
    """Test Jensen's alpha when returns exactly match CAPM prediction."""
    # Construct returns to exactly match CAPM with beta = 1
    # Portfolio returns = rfr + beta * (bmk - rfr) = rfr + 1 * (bmk - rfr) = bmk
    # So if portfolio = benchmark, alpha should be 0
    returns = pd.Series([0.02, 0.04, 0.03])
    bmk = pd.Series([0.02, 0.04, 0.03])
    rfr = pd.Series([0.01, 0.01, 0.01])

    result = JensensAlpha._summary(returns=returns, bmk=bmk, rfr=rfr, ann_factor=252)

    # Alpha should be near zero (matches CAPM with beta=1)
    assert np.isclose(result, 0, atol=0.01)
