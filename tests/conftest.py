"""Test fixtures for portfolio assessment tests."""

import pandas as pd
import pytest


@pytest.fixture
def simple_returns():
    """Simple returns: 1%, 2%, 3%"""
    return pd.Series([0.01, 0.02, 0.03])


@pytest.fixture
def simple_benchmark():
    """Simple benchmark returns: 0.5%, 1%, 1.5%"""
    return pd.Series([0.005, 0.01, 0.015])


@pytest.fixture
def simple_rfr():
    """Simple risk-free rate: 0.1% daily"""
    return pd.Series([0.001, 0.001, 0.001])


@pytest.fixture
def correlated_returns():
    """Returns perfectly correlated with benchmark: 2%, 4%, 8%"""
    return pd.Series([0.02, 0.04, 0.08])


@pytest.fixture
def correlated_benchmark():
    """Benchmark returns: 1%, 2%, 4% (perfect correlation)"""
    return pd.Series([0.01, 0.02, 0.04])


@pytest.fixture
def positive_returns():
    """All positive returns: 2%, 3%, 4%"""
    return pd.Series([0.02, 0.03, 0.04])


@pytest.fixture
def negative_returns():
    """All negative returns: -1%, -2%, -3%"""
    return pd.Series([-0.01, -0.02, -0.03])


@pytest.fixture
def mixed_returns():
    """Mixed returns: -2%, 0%, 2%"""
    return pd.Series([-0.02, 0.00, 0.02])


@pytest.fixture
def volatile_returns():
    """Volatile returns: -5%, 5%, -5%, 5%"""
    return pd.Series([-0.05, 0.05, -0.05, 0.05])


@pytest.fixture
def zero_returns():
    """Zero returns"""
    return pd.Series([0.0, 0.0, 0.0])


@pytest.fixture
def constant_returns():
    """Constant 1% returns"""
    return pd.Series([0.01, 0.01, 0.01])


@pytest.fixture
def drawdown_returns():
    """Returns that create drawdown: 10%, -5%, -5%, 5%"""
    return pd.Series([0.10, -0.05, -0.05, 0.05])


@pytest.fixture
def skewed_returns():
    """Right-skewed returns: 1%, 1%, 1%, 1%, 10%"""
    return pd.Series([0.01, 0.01, 0.01, 0.01, 0.10])


@pytest.fixture
def symmetric_returns():
    """Symmetric returns around zero: -2%, -1%, 0%, 1%, 2%"""
    return pd.Series([-0.02, -0.01, 0.00, 0.01, 0.02])
