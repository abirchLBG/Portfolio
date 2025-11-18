"""Tests for Omega Ratio assessment."""

import numpy as np
import pandas as pd

from src.assessments.omega_ratio import OmegaRatio


def test_omega_ratio_all_gains():
    """Test Omega ratio when all returns are above threshold."""
    # Returns: 2%, 3%, 4% all above 0% threshold
    # Gains = 2% + 3% + 4% = 9%
    # Losses = 0
    # Omega = inf
    returns = pd.Series([0.02, 0.03, 0.04])

    result = OmegaRatio._summary(returns=returns, threshold=0.0, ann_factor=252)

    # All gains, no losses, should be inf
    assert np.isinf(result)


def test_omega_ratio_mixed_returns():
    """Test Omega ratio with both gains and losses."""
    # Returns: -2%, 0%, 4% with threshold = 0%
    # Gains: 4%
    # Losses: 2%
    # Omega = 4% / 2% = 2.0
    returns = pd.Series([-0.02, 0.00, 0.04])

    result = OmegaRatio._summary(returns=returns, threshold=0.0, ann_factor=252)

    # Gains / Losses = 4 / 2 = 2.0
    assert np.isclose(result, 2.0, rtol=0.01)


def test_omega_ratio_all_losses():
    """Test Omega ratio when all returns are below threshold."""
    # Returns: -3%, -2%, -1% all below 0% threshold
    # Gains = 0
    # Losses = 6%
    # Omega = 0
    returns = pd.Series([-0.03, -0.02, -0.01])

    result = OmegaRatio._summary(returns=returns, threshold=0.0, ann_factor=252)

    # No gains, all losses, should be 0
    assert result == 0.0


def test_omega_ratio_nonzero_threshold():
    """Test Omega ratio with non-zero threshold."""
    # Returns: 1%, 2%, 3% with threshold = 2% (daily = 2/252 ≈ 0.00794%)
    # Above threshold: 3% - threshold
    # Below threshold: threshold - 1%, threshold - 2%
    returns = pd.Series([0.01, 0.02, 0.03])

    result = OmegaRatio._summary(returns=returns, threshold=0.02, ann_factor=252)

    # Should be positive but less than inf
    assert result > 0 and not np.isinf(result)
