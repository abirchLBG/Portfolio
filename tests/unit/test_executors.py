"""Tests for executor types."""

import pytest
from unittest.mock import Mock, patch
from concurrent.futures import ProcessPoolExecutor

from src.utils.executors import DummyExecutor, DummyFuture, APIFuture, RQExecutor


class TestDummyFuture:
    def test_successful_result(self):
        """Test DummyFuture with successful function execution."""

        def add(a, b):
            return a + b

        future = DummyFuture(add, (1, 2), {})
        assert future.result() == 3

    def test_exception_handling(self):
        """Test DummyFuture with function that raises exception."""

        def error_fn():
            raise ValueError("Test error")

        future = DummyFuture(error_fn, (), {})
        with pytest.raises(ValueError, match="Test error"):
            future.result()

    def test_with_kwargs(self):
        """Test DummyFuture with keyword arguments."""

        def multiply(a, b=2):
            return a * b

        future = DummyFuture(multiply, (5,), {"b": 3})
        assert future.result() == 15


class TestDummyExecutor:
    def test_submit(self):
        """Test DummyExecutor submit method."""
        executor = DummyExecutor()

        def add(a, b):
            return a + b

        future = executor.submit(add, 10, 20)
        assert future.result() == 30

    def test_shutdown(self):
        """Test DummyExecutor shutdown method."""
        executor = DummyExecutor()
        executor.shutdown()  # Should not raise

    def test_shutdown_with_wait(self):
        """Test DummyExecutor shutdown with wait parameter."""
        executor = DummyExecutor()
        executor.shutdown(wait=True)  # Should not raise
        executor.shutdown(wait=False)  # Should not raise


class TestAPIFuture:
    @patch("src.utils.executors.requests.get")
    def test_successful_result(self, mock_get):
        """Test APIFuture polling for successful result."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": "finished",
            "result": {"value": 42},
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        future = APIFuture("job123", "http://api.example.com", poll_interval=0.01)
        result = future.result()

        assert result == {"value": 42}
        mock_get.assert_called_with("http://api.example.com/status/job123")

    @patch("src.utils.executors.requests.get")
    def test_failed_job(self, mock_get):
        """Test APIFuture polling for failed job."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "failed"}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        future = APIFuture("job123", "http://api.example.com", poll_interval=0.01)

        with pytest.raises(RuntimeError, match="Job job123 failed"):
            future.result()

    @patch("src.utils.executors.requests.get")
    @patch("src.utils.executors.time.sleep")
    def test_timeout(self, mock_sleep, mock_get):
        """Test APIFuture timeout."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "running"}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        future = APIFuture("job123", "http://api.example.com", poll_interval=0.01)

        with pytest.raises(TimeoutError, match="Job job123 timed out"):
            future.result(timeout=0.1)

    @patch("src.utils.executors.requests.get")
    @patch("src.utils.executors.time.sleep")
    def test_polling_until_finished(self, mock_sleep, mock_get):
        """Test APIFuture polls multiple times until finished."""
        responses = [
            {"status": "queued"},
            {"status": "running"},
            {"status": "finished", "result": "done"},
        ]

        mock_response = Mock()
        mock_response.json.side_effect = responses
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        future = APIFuture("job123", "http://api.example.com", poll_interval=0.01)
        result = future.result()

        assert result == "done"
        assert mock_get.call_count == 3

    @patch("src.utils.executors.requests.get")
    def test_result_is_none(self, mock_get):
        """Test APIFuture when result is None but status is finished."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "finished", "result": None}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        future = APIFuture("job123", "http://api.example.com", poll_interval=0.01)

        with pytest.raises(RuntimeError, match="finished but result is None"):
            future.result()

    def test_url_normalization(self):
        """Test that trailing slash is removed from URL."""
        future = APIFuture("job123", "http://api.example.com/", poll_interval=0.01)
        assert future.api_url == "http://api.example.com"


class TestRQExecutor:
    @patch("src.utils.executors.requests.post")
    def test_submit(self, mock_post):
        """Test RQExecutor submit method."""
        import pandas as pd
        from src.assessments.beta import Beta
        from src.dataclasses.assessment_config import AssessmentConfig

        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {"job_id": "job123"}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        # Create assessment with enough data points (need at least min_periods)
        config = AssessmentConfig(
            returns=pd.Series([0.01 + i * 0.001 for i in range(25)]),
            bmk=pd.Series([0.005 + i * 0.0005 for i in range(25)]),
            rfr=pd.Series([0.001] * 25),
            min_periods=2,
        )
        assessment = Beta(config=config)

        # Submit job
        executor = RQExecutor("http://api.example.com", poll_interval=0.01)
        future = executor.submit(assessment._run, "summary")

        # Verify POST request
        assert isinstance(future, APIFuture)
        assert future.job_id == "job123"
        mock_post.assert_called_once()

        # Verify payload
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://api.example.com/run"
        payload = call_args[1]["json"]
        assert payload["assessment_name"] == "Beta"
        assert payload["assessment_type"] == "summary"
        assert "returns" in payload["config"]
        assert len(payload["config"]["returns"]) == 25

    @patch("src.utils.executors.requests.post")
    def test_submit_serializes_config(self, mock_post):
        """Test RQExecutor properly serializes config."""
        import pandas as pd
        from src.assessments.beta import Beta
        from src.dataclasses.assessment_config import AssessmentConfig

        mock_response = Mock()
        mock_response.json.return_value = {"job_id": "job123"}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        # Create config with numpy arrays
        config = AssessmentConfig(
            returns=pd.Series([0.01 + i * 0.001 for i in range(25)]),
            bmk=pd.Series([0.005 + i * 0.0005 for i in range(25)]),
            rfr=pd.Series([0.001] * 25),
            min_periods=2,
            ann_factor=252,
        )
        assessment = Beta(config=config)

        executor = RQExecutor("http://api.example.com")
        executor.submit(assessment._run, "summary")

        # Verify config is serialized to lists
        payload = mock_post.call_args[1]["json"]
        assert isinstance(payload["config"]["returns"], list)
        assert isinstance(payload["config"]["bmk"], list)
        assert isinstance(payload["config"]["rfr"], list)
        assert payload["config"]["ann_factor"] == 252

    def test_shutdown(self):
        """Test RQExecutor shutdown method."""
        executor = RQExecutor("http://api.example.com")
        executor.shutdown()  # Should not raise

    def test_url_normalization(self):
        """Test that trailing slash is removed from URL."""
        executor = RQExecutor("http://api.example.com/")
        assert executor.api_url == "http://api.example.com"


def _square_function(x):
    """Helper function for ProcessPoolExecutor test (must be at module level for pickling)."""
    return x**2


class TestExecutorIntegration:
    def test_dummy_executor_with_multiple_tasks(self):
        """Test DummyExecutor with multiple concurrent tasks."""
        executor = DummyExecutor()
        futures = []

        for i in range(5):
            future = executor.submit(lambda x: x * 2, i)
            futures.append(future)

        results = [f.result() for f in futures]
        assert results == [0, 2, 4, 6, 8]

    def test_process_pool_executor_basic(self):
        """Test ProcessPoolExecutor basic functionality."""
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(_square_function, i) for i in range(5)]
            results = [f.result() for f in futures]

        assert results == [0, 1, 4, 9, 16]
