"""Tests for Down Capture Ratio assessment."""

import numpy as np
import pandas as pd

from src.assessments.down_capture import DownCapture


def test_down_capture_better_protection():
    """Test down capture when portfolio loses less in down markets."""
    # Down days: portfolio loses less than benchmark
    # Returns: 2%, -1%, -2%
    # Benchmark: 1%, -2%, -4%
    # Down days: (-1%, -2%) vs (-2%, -4%)
    # Down capture = (mean -1.5%) / (mean -3%) = 0.5 (50%)
    returns = pd.Series([0.02, -0.01, -0.02])
    bmk = pd.Series([0.01, -0.02, -0.04])

    result = DownCapture._summary(returns=returns, bmk=bmk)

    # Should be < 100% (losing less in down markets - good)
    assert result < 1.0


def test_down_capture_worse_protection():
    """Test down capture when portfolio loses more in down markets."""
    # Returns: 1%, -4%, -2%
    # Benchmark: 2%, -2%, -1%
    # Down days: (-4%, -2%) vs (-2%, -1%)
    # Down capture = (mean -3%) / (mean -1.5%) = 2.0 (200%)
    returns = pd.Series([0.01, -0.04, -0.02])
    bmk = pd.Series([0.02, -0.02, -0.01])

    result = DownCapture._summary(returns=returns, bmk=bmk)

    # Should be > 100% (losing more in down markets - bad)
    assert result > 1.0


def test_down_capture_matching():
    """Test down capture when portfolio matches benchmark in down markets."""
    # Same returns on down days
    returns = pd.Series([0.02, -0.01, -0.02])
    bmk = pd.Series([0.01, -0.01, -0.02])

    result = DownCapture._summary(returns=returns, bmk=bmk)

    # Should be close to 100%
    assert np.isclose(result, 1.0, rtol=0.01)


def test_down_capture_no_down_days():
    """Test down capture when there are no down days."""
    # All positive benchmark returns
    returns = pd.Series([0.01, 0.02, 0.03])
    bmk = pd.Series([0.01, 0.02, 0.03])

    result = DownCapture._summary(returns=returns, bmk=bmk)

    # Should be NaN (no down days to measure)
    assert np.isnan(result)
