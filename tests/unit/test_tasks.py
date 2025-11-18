"""Tests for RQ tasks."""

import pytest
import pandas as pd

from src.app.tasks import add_numbers, run_assessment


class TestAddNumbers:
    def test_add_numbers(self):
        """Test add_numbers task."""
        result = add_numbers(5, 3)
        assert result == 8

    def test_add_negative_numbers(self):
        """Test add_numbers with negative numbers."""
        result = add_numbers(-5, 3)
        assert result == -2


class TestRunAssessment:
    def test_run_assessment_summary(self):
        """Test run_assessment with summary type."""
        # Need at least 21 data points for min_periods
        config = {
            "returns": [0.01 + i * 0.001 for i in range(25)],
            "bmk": [0.005 + i * 0.0005 for i in range(25)],
            "rfr": [0.001] * 25,
            "min_periods": 2,
        }

        result = run_assessment("Beta", "summary", config)

        assert "result" in result
        assert "time" in result
        assert isinstance(result["result"], float)

    def test_run_assessment_rolling(self):
        """Test run_assessment with rolling type."""
        config = {
            "returns": [0.01 + i * 0.001 for i in range(25)],
            "bmk": [0.005 + i * 0.0005 for i in range(25)],
            "rfr": [0.001] * 25,
            "window": 5,
            "min_periods": 3,
        }

        result = run_assessment("Beta", "rolling", config)

        assert "result" in result
        assert isinstance(result["result"], pd.Series)

    def test_run_assessment_expanding(self):
        """Test run_assessment with expanding type."""
        config = {
            "returns": [0.01 + i * 0.001 for i in range(25)],
            "bmk": [0.005 + i * 0.0005 for i in range(25)],
            "rfr": [0.001] * 25,
            "min_periods": 2,
        }

        result = run_assessment("Beta", "expanding", config)

        assert "result" in result
        assert isinstance(result["result"], pd.Series)

    def test_run_assessment_invalid_name(self):
        """Test run_assessment with invalid assessment name."""
        config = {
            "returns": [0.01 + i * 0.001 for i in range(25)],
            "bmk": [0.005 + i * 0.0005 for i in range(25)],
            "rfr": [0.001] * 25,
            "min_periods": 2,
        }

        with pytest.raises(ValueError, match="Unknown assessment"):
            run_assessment("InvalidAssessment", "summary", config)

    def test_run_assessment_with_optional_params(self):
        """Test run_assessment with optional parameters."""
        config = {
            "returns": [0.01 + i * 0.001 for i in range(25)],
            "bmk": [0.005 + i * 0.0005 for i in range(25)],
            "rfr": [0.001] * 25,
            "window": 5,
            "min_periods": 3,
            "ann_factor": 252,
        }

        result = run_assessment("Beta", "rolling", config)
        assert "result" in result

    def test_run_assessment_preserves_config(self):
        """Test that run_assessment doesn't mutate original config."""
        config = {
            "returns": [0.01 + i * 0.001 for i in range(25)],
            "bmk": [0.005 + i * 0.0005 for i in range(25)],
            "rfr": [0.001] * 25,
            "min_periods": 2,
        }
        original_config = config.copy()

        run_assessment("Beta", "summary", config)

        # Original config should be unchanged
        assert config == original_config

    def test_run_assessment_different_assessments(self):
        """Test run_assessment with different assessment types."""
        config = {
            "returns": [0.01 + i * 0.001 for i in range(25)],
            "bmk": [0.005 + i * 0.0005 for i in range(25)],
            "rfr": [0.001] * 25,
            "min_periods": 2,
        }

        assessments = ["Beta", "SharpeRatio", "Volatility", "Correlation"]
        for assessment_name in assessments:
            result = run_assessment(assessment_name, "summary", config)
            assert "result" in result
            assert "time" in result
