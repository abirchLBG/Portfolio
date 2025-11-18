"""Tests for Kurtosis assessment."""

import numpy as np
import pandas as pd

from src.assessments.kurtosis import Kurtosis


def test_kurtosis_normal_like():
    """Test kurtosis with normal-like distribution."""
    # Random normal-ish returns
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.01, 0.02, 100))

    result = Kurtosis._summary(returns=returns, excess=True)

    # Excess kurtosis should be near 0 for normal distribution
    # Allow wider range due to sample size
    assert -1 < result < 1


def test_kurtosis_fat_tails():
    """Test kurtosis with fat tails (leptokurtic)."""
    # Distribution with outliers: mostly small, few large
    returns = pd.Series([0.01] * 10 + [-0.10, 0.10])

    result = Kurtosis._summary(returns=returns, excess=True)

    # Should be positive (fat tails, more extreme values)
    assert result > 0


def test_kurtosis_thin_tails():
    """Test kurtosis with thin tails (platykurtic)."""
    # Uniform-ish distribution: -1%, 0%, 1% with equal frequency
    returns = pd.Series([-0.01, -0.01, 0.00, 0.00, 0.01, 0.01])

    result = Kurtosis._summary(returns=returns, excess=True)

    # Should be negative (thinner tails than normal)
    assert result < 0


def test_kurtosis_excess_vs_raw():
    """Test excess vs raw kurtosis."""
    returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])

    excess = Kurtosis._summary(returns=returns, excess=True)
    raw = Kurtosis._summary(returns=returns, excess=False)

    # Raw = Excess + 3
    assert np.isclose(raw, excess + 3, rtol=0.01)
