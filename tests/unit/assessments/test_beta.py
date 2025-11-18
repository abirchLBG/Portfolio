"""Tests for Beta assessment."""

import numpy as np
import pandas as pd

from src.assessments.beta import Beta


def test_beta_perfect_correlation():
    """Test beta with perfect correlation (beta = 2)."""
    # Returns: 2%, 4%, 8%
    # Benchmark: 1%, 2%, 4%
    # Returns are exactly 2x benchmark, so beta = 2
    returns = pd.Series([0.02, 0.04, 0.08])
    bmk = pd.Series([0.01, 0.02, 0.04])

    result = Beta._summary(returns=returns, bmk=bmk)

    # Beta should be 2.0
    assert np.isclose(result, 2.0, rtol=0.01)


def test_beta_market_portfolio():
    """Test beta when portfolio matches market (beta = 1)."""
    # Identical returns
    returns = pd.Series([0.01, 0.02, 0.03])
    bmk = pd.Series([0.01, 0.02, 0.03])

    result = Beta._summary(returns=returns, bmk=bmk)

    # Beta should be 1.0
    assert np.isclose(result, 1.0, rtol=0.01)


def test_beta_defensive_portfolio():
    """Test beta for defensive portfolio (beta < 1)."""
    # Returns: 0.5%, 1%, 1.5%
    # Benchmark: 1%, 2%, 3%
    # Returns are 0.5x benchmark, so beta = 0.5
    returns = pd.Series([0.005, 0.01, 0.015])
    bmk = pd.Series([0.01, 0.02, 0.03])

    result = Beta._summary(returns=returns, bmk=bmk)

    # Beta should be around 0.5
    assert 0.4 < result < 0.6


def test_beta_negative():
    """Test beta with negative correlation."""
    # Returns move opposite to benchmark
    returns = pd.Series([0.03, 0.02, 0.01])
    bmk = pd.Series([0.01, 0.02, 0.03])

    result = Beta._summary(returns=returns, bmk=bmk)

    # Beta should be negative
    assert result < 0
