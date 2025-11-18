"""Tests for Up Capture Ratio assessment."""

import numpy as np
import pandas as pd

from src.assessments.up_capture import UpCapture


def test_up_capture_outperformance():
    """Test up capture when portfolio outperforms in up markets."""
    # Up days: portfolio gains more than benchmark
    # Returns: 2%, 4%, -1%
    # Benchmark: 1%, 2%, -2%
    # Up days: (2%, 4%) vs (1%, 2%)
    # Up capture = (mean 3%) / (mean 1.5%) = 2.0 (200%)
    returns = pd.Series([0.02, 0.04, -0.01])
    bmk = pd.Series([0.01, 0.02, -0.02])

    result = UpCapture._summary(returns=returns, bmk=bmk)

    # Should be > 100% (outperforming in up markets)
    assert result > 1.0


def test_up_capture_underperformance():
    """Test up capture when portfolio underperforms in up markets."""
    # Returns: 1%, 2%, -1%
    # Benchmark: 2%, 4%, -2%
    # Up days: (1%, 2%) vs (2%, 4%)
    # Up capture = (mean 1.5%) / (mean 3%) = 0.5 (50%)
    returns = pd.Series([0.01, 0.02, -0.01])
    bmk = pd.Series([0.02, 0.04, -0.02])

    result = UpCapture._summary(returns=returns, bmk=bmk)

    # Should be < 100% (underperforming in up markets)
    assert result < 1.0


def test_up_capture_matching():
    """Test up capture when portfolio matches benchmark in up markets."""
    # Same returns on up days
    returns = pd.Series([0.02, 0.03, -0.01])
    bmk = pd.Series([0.02, 0.03, -0.02])

    result = UpCapture._summary(returns=returns, bmk=bmk)

    # Should be close to 100%
    assert np.isclose(result, 1.0, rtol=0.01)


def test_up_capture_no_up_days():
    """Test up capture when there are no up days."""
    # All negative benchmark returns
    returns = pd.Series([-0.01, -0.02, -0.03])
    bmk = pd.Series([-0.01, -0.02, -0.03])

    result = UpCapture._summary(returns=returns, bmk=bmk)

    # Should be NaN (no up days to measure)
    assert np.isnan(result)
