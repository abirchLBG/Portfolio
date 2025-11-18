"""Tests for Evaluation class."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from concurrent.futures import ProcessPoolExecutor

from src.evaluation import (
    Evaluation,
    ExecutorType,
    ALL_ASSESSMENTS,
    ALL_ASSESSMENT_TYPES,
)
from src.dataclasses.assessment_config import AssessmentConfig
from src.dataclasses.assessment_results import AssessmentType
from src.constants import AssessmentName
from src.utils.executors import DummyExecutor, RQExecutor


@pytest.fixture
def sample_config():
    """Create sample assessment config."""
    # Create data with at least min_periods (21) observations
    returns = pd.Series([0.01 + i * 0.001 for i in range(30)])
    bmk = pd.Series([0.005 + i * 0.0005 for i in range(30)])
    rfr = pd.Series([0.001] * 30)

    return AssessmentConfig(
        returns=returns,
        bmk=bmk,
        rfr=rfr,
        min_periods=21,
        window=252,
    )


class TestExecutorType:
    def test_executor_type_default(self):
        """Test ExecutorType.DEFAULT creates DummyExecutor."""
        executor = ExecutorType.DEFAULT()
        assert isinstance(executor, DummyExecutor)

    def test_executor_type_process_pool(self):
        """Test ExecutorType.ProcessPool creates ProcessPoolExecutor."""
        executor = ExecutorType.ProcessPool(max_workers=2)
        assert isinstance(executor, ProcessPoolExecutor)
        executor.shutdown()

    def test_executor_type_remote(self):
        """Test ExecutorType.Remote creates RQExecutor."""
        executor = ExecutorType.Remote(api_url="http://localhost:8000")
        assert isinstance(executor, RQExecutor)


class TestEvaluation:
    def test_initialization(self, sample_config):
        """Test Evaluation initialization."""
        eval_obj = Evaluation(config=sample_config)
        assert eval_obj.config == sample_config
        assert eval_obj._assessments == ALL_ASSESSMENTS
        assert eval_obj._assessment_types == list(AssessmentType)
        assert isinstance(eval_obj._executor, DummyExecutor)

    def test_repr(self, sample_config):
        """Test Evaluation __repr__."""
        eval_obj = Evaluation(config=sample_config)
        assert repr(eval_obj) == "Evaluation"

    def test_with_assessments_filter(self, sample_config):
        """Test filtering assessments."""
        eval_obj = Evaluation(config=sample_config)
        filtered_assessments = [AssessmentName.Beta, AssessmentName.SharpeRatio]

        result = eval_obj.with_assessments(filtered_assessments)

        assert result is eval_obj  # Check fluent interface
        assert len(eval_obj._assessments) == 2
        assert AssessmentName.Beta in eval_obj._assessments
        assert AssessmentName.SharpeRatio in eval_obj._assessments

    def test_with_assessments_none(self, sample_config):
        """Test with_assessments with None returns same object."""
        eval_obj = Evaluation(config=sample_config)
        result = eval_obj.with_assessments(None)

        assert result is eval_obj
        assert eval_obj._assessments == ALL_ASSESSMENTS

    def test_with_assessments_empty(self, sample_config):
        """Test with_assessments with empty list returns same object."""
        eval_obj = Evaluation(config=sample_config)
        result = eval_obj.with_assessments([])

        assert result is eval_obj
        assert eval_obj._assessments == ALL_ASSESSMENTS

    def test_with_assessment_types_filter(self, sample_config):
        """Test filtering assessment types."""
        eval_obj = Evaluation(config=sample_config)
        filtered_types = [AssessmentType.Summary, AssessmentType.Rolling]

        result = eval_obj.with_assessment_types(filtered_types)

        assert result is eval_obj
        assert len(eval_obj._assessment_types) == 2
        assert AssessmentType.Summary in eval_obj._assessment_types
        assert AssessmentType.Rolling in eval_obj._assessment_types

    def test_with_assessment_types_none(self, sample_config):
        """Test with_assessment_types with None."""
        eval_obj = Evaluation(config=sample_config)
        result = eval_obj.with_assessment_types(None)

        assert result is eval_obj
        assert eval_obj._assessment_types == list(AssessmentType)

    def test_with_executor_dummy(self, sample_config):
        """Test setting DummyExecutor."""
        eval_obj = Evaluation(config=sample_config)
        executor = DummyExecutor()

        result = eval_obj.with_executor(executor)

        assert result is eval_obj
        assert eval_obj._executor is executor

    def test_with_executor_process_pool(self, sample_config):
        """Test setting ProcessPoolExecutor."""
        eval_obj = Evaluation(config=sample_config)
        executor = ProcessPoolExecutor(max_workers=2)

        result = eval_obj.with_executor(executor)

        assert result is eval_obj
        assert eval_obj._executor is executor
        executor.shutdown()

    def test_with_executor_remote(self, sample_config):
        """Test setting RQExecutor."""
        eval_obj = Evaluation(config=sample_config)
        executor = RQExecutor(api_url="http://localhost:8000")

        result = eval_obj.with_executor(executor)

        assert result is eval_obj
        assert eval_obj._executor is executor

    def test_init_assessments(self, sample_config):
        """Test assessment initialization."""
        eval_obj = Evaluation(config=sample_config)
        eval_obj._init_assessments()

        assert hasattr(eval_obj, "_initialized_assessments")
        assert len(eval_obj._initialized_assessments) == len(ALL_ASSESSMENTS)

        # Check that all assessments are initialized with config
        for name, assessment in eval_obj._initialized_assessments.items():
            assert assessment.config == sample_config

    def test_run_with_dummy_executor(self, sample_config):
        """Test run with DummyExecutor."""
        eval_obj = (
            Evaluation(config=sample_config)
            .with_assessments([AssessmentName.Beta])
            .with_assessment_types([AssessmentType.Summary])
        )

        results = eval_obj.run()

        assert AssessmentName.Beta in results.results
        assert AssessmentType.Summary in results.results[AssessmentName.Beta]
        assert isinstance(
            results.results[AssessmentName.Beta][AssessmentType.Summary], float
        )
        assert AssessmentName.Beta in results.timer
        assert AssessmentType.Summary in results.timer[AssessmentName.Beta]

    def test_run_with_process_pool_executor(self, sample_config):
        """Test run with ProcessPoolExecutor."""
        executor = ProcessPoolExecutor(max_workers=2)
        eval_obj = (
            Evaluation(config=sample_config)
            .with_assessments([AssessmentName.Beta, AssessmentName.SharpeRatio])
            .with_assessment_types([AssessmentType.Summary])
            .with_executor(executor)
        )

        results = eval_obj.run()

        assert AssessmentName.Beta in results.results
        assert AssessmentName.SharpeRatio in results.results
        assert AssessmentType.Summary in results.results[AssessmentName.Beta]
        executor.shutdown()

    def test_run_with_rolling_and_expanding(self, sample_config):
        """Test run with rolling and expanding types."""
        eval_obj = (
            Evaluation(config=sample_config)
            .with_assessments([AssessmentName.Beta])
            .with_assessment_types([AssessmentType.Rolling, AssessmentType.Expanding])
        )

        results = eval_obj.run()

        assert AssessmentName.Beta in results.results
        assert AssessmentType.Rolling in results.results[AssessmentName.Beta]
        assert AssessmentType.Expanding in results.results[AssessmentName.Beta]
        assert isinstance(
            results.results[AssessmentName.Beta][AssessmentType.Rolling], pd.Series
        )
        assert isinstance(
            results.results[AssessmentName.Beta][AssessmentType.Expanding], pd.Series
        )

    def test_run_multiple_assessments(self, sample_config):
        """Test run with multiple assessments."""
        assessments = [
            AssessmentName.Beta,
            AssessmentName.SharpeRatio,
            AssessmentName.Volatility,
        ]
        eval_obj = Evaluation(config=sample_config).with_assessments(assessments)

        results = eval_obj.run()

        for assessment in assessments:
            assert assessment in results.results
            for assessment_type in eval_obj._assessment_types:
                assert assessment_type in results.results[assessment]

    def test_run_all_assessment_types(self, sample_config):
        """Test run with all assessment types."""
        eval_obj = Evaluation(config=sample_config).with_assessments(
            [AssessmentName.Beta]
        )

        results = eval_obj.run()

        assert AssessmentName.Beta in results.results
        for assessment_type in AssessmentType:
            assert assessment_type in results.results[AssessmentName.Beta]

    @patch("src.utils.executors.requests.post")
    def test_run_with_rq_executor(self, mock_post, sample_config):
        """Test run with RQExecutor (mocked)."""
        # Mock the POST request to enqueue jobs
        mock_response = Mock()
        mock_response.json.return_value = {"job_id": "job123"}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        # Mock the GET request to check status
        with patch("src.utils.executors.requests.get") as mock_get:
            mock_get_response = Mock()
            mock_get_response.json.return_value = {
                "status": "finished",
                "result": {"result": 1.5, "time": 0.001},
            }
            mock_get_response.raise_for_status = Mock()
            mock_get.return_value = mock_get_response

            executor = RQExecutor(api_url="http://localhost:8000", poll_interval=0.01)
            eval_obj = (
                Evaluation(config=sample_config)
                .with_assessments([AssessmentName.Beta])
                .with_assessment_types([AssessmentType.Summary])
                .with_executor(executor)
            )

            results = eval_obj.run()

            assert AssessmentName.Beta in results.results
            assert mock_post.called

    def test_fluent_interface(self, sample_config):
        """Test fluent interface chaining."""
        executor = DummyExecutor()
        eval_obj = (
            Evaluation(config=sample_config)
            .with_assessments([AssessmentName.Beta])
            .with_assessment_types([AssessmentType.Summary])
            .with_executor(executor)
        )

        assert isinstance(eval_obj, Evaluation)
        assert len(eval_obj._assessments) == 1
        assert len(eval_obj._assessment_types) == 1
        assert eval_obj._executor is executor

    def test_run_timing_data(self, sample_config):
        """Test that timing data is captured."""
        eval_obj = (
            Evaluation(config=sample_config)
            .with_assessments([AssessmentName.Beta])
            .with_assessment_types([AssessmentType.Summary])
        )

        results = eval_obj.run()

        assert AssessmentName.Beta in results.timer
        assert AssessmentType.Summary in results.timer[AssessmentName.Beta]
        assert isinstance(
            results.timer[AssessmentName.Beta][AssessmentType.Summary], float
        )
        assert results.timer[AssessmentName.Beta][AssessmentType.Summary] > 0


class TestAllAssessments:
    def test_all_assessments_has_implementations(self):
        """Test ALL_ASSESSMENTS has implementations for registered assessments."""
        # Check that all items in ALL_ASSESSMENTS are valid
        assert len(ALL_ASSESSMENTS) > 0
        for name, assessment_class in ALL_ASSESSMENTS.items():
            assert isinstance(name, AssessmentName)
            assert callable(assessment_class)

    def test_all_assessment_types_frozen(self):
        """Test ALL_ASSESSMENT_TYPES is a frozenset."""
        assert isinstance(ALL_ASSESSMENT_TYPES, frozenset)
        assert len(ALL_ASSESSMENT_TYPES) == len(AssessmentType)
