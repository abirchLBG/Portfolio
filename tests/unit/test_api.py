"""Tests for FastAPI endpoints."""

import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np
import sys

# Create mock task_queue module
mock_task_queue_module = Mock()
mock_redis = Mock()
mock_redis.ping.return_value = True
mock_queue = Mock()
mock_task_queue_module.redis_conn = mock_redis
mock_task_queue_module.task_queue = mock_queue
mock_task_queue_module.get_redis_connection.return_value = mock_redis

# Inject mock before importing API module
sys.modules["src.app.task_queue"] = mock_task_queue_module

from src.app.api import app, AssessmentRequest, serialize_result  # noqa: E402


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_job():
    """Create mock RQ job."""
    job = Mock()
    job.id = "test_job_123"
    job.get_id.return_value = "test_job_123"
    job.get_status.return_value = "finished"
    job.is_finished = True
    job.result = {"value": 42}
    return job


class TestPingEndpoint:
    def test_ping(self, client):
        """Test ping endpoint returns pong."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == "pong"


class TestAddEndpoint:
    @patch("src.app.api.task_queue")
    def test_enqueue_add(self, mock_queue, client, mock_job):
        """Test add endpoint enqueues task."""
        mock_queue.enqueue.return_value = mock_job

        response = client.post("/add/?a=5&b=3")
        assert response.status_code == 200
        assert response.json() == {"job_id": "test_job_123"}

        mock_queue.enqueue.assert_called_once()


class TestStatusEndpoint:
    @patch("src.app.api.Job")
    @patch("src.app.api.task_queue")
    def test_get_status_finished(self, mock_queue, mock_job_class, client):
        """Test status endpoint for finished job."""
        job = Mock()
        job.id = "job123"
        job.get_status.return_value = "finished"
        job.is_finished = True
        job.result = {"value": 42}
        mock_job_class.fetch.return_value = job

        response = client.get("/status/job123")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "job123"
        assert data["status"] == "finished"
        assert data["result"] == {"value": 42}

    @patch("src.app.api.Job")
    @patch("src.app.api.task_queue")
    def test_get_status_running(self, mock_queue, mock_job_class, client):
        """Test status endpoint for running job."""
        job = Mock()
        job.id = "job123"
        job.get_status.return_value = "started"
        job.is_finished = False
        job.result = None
        mock_job_class.fetch.return_value = job

        response = client.get("/status/job123")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert data["result"] is None


class TestAssessmentRequest:
    def test_valid_request(self):
        """Test valid AssessmentRequest."""
        req = AssessmentRequest(
            assessment_name="Beta",
            assessment_type="summary",
            config={
                "returns": [0.01, 0.02],
                "bmk": [0.005, 0.01],
                "rfr": [0.001, 0.001],
            },
        )
        assert req.assessment_name == "Beta"
        assert req.assessment_type == "summary"

    def test_invalid_assessment_name(self):
        """Test invalid assessment name."""
        with pytest.raises(ValueError, match="Invalid assessment_name"):
            AssessmentRequest(
                assessment_name="InvalidAssessment",
                assessment_type="summary",
                config={
                    "returns": [0.01, 0.02],
                    "bmk": [0.005, 0.01],
                    "rfr": [0.001, 0.001],
                },
            )

    def test_invalid_assessment_type(self):
        """Test invalid assessment type."""
        with pytest.raises(ValueError, match="Invalid assessment_type"):
            AssessmentRequest(
                assessment_name="Beta",
                assessment_type="invalid_type",
                config={
                    "returns": [0.01, 0.02],
                    "bmk": [0.005, 0.01],
                    "rfr": [0.001, 0.001],
                },
            )

    def test_missing_required_config_fields(self):
        """Test missing required config fields."""
        with pytest.raises(ValueError, match="Config missing required fields"):
            AssessmentRequest(
                assessment_name="Beta",
                assessment_type="summary",
                config={"returns": [0.01, 0.02]},
            )

    def test_config_field_not_list(self):
        """Test config field that is not a list."""
        with pytest.raises(ValueError, match="must be a list"):
            AssessmentRequest(
                assessment_name="Beta",
                assessment_type="summary",
                config={
                    "returns": "not a list",
                    "bmk": [0.005, 0.01],
                    "rfr": [0.001, 0.001],
                },
            )

    def test_config_empty_list(self):
        """Test config with empty list."""
        with pytest.raises(ValueError, match="cannot be empty"):
            AssessmentRequest(
                assessment_name="Beta",
                assessment_type="summary",
                config={"returns": [], "bmk": [0.005, 0.01], "rfr": [0.001, 0.001]},
            )

    def test_config_invalid_numeric_param(self):
        """Test config with invalid numeric parameter."""
        with pytest.raises(ValueError, match="must be numeric"):
            AssessmentRequest(
                assessment_name="Beta",
                assessment_type="rolling",
                config={
                    "returns": [0.01, 0.02],
                    "bmk": [0.005, 0.01],
                    "rfr": [0.001, 0.001],
                    "window": "not a number",
                },
            )

    def test_config_negative_window(self):
        """Test config with negative window."""
        with pytest.raises(ValueError, match="must be positive"):
            AssessmentRequest(
                assessment_name="Beta",
                assessment_type="rolling",
                config={
                    "returns": [0.01, 0.02],
                    "bmk": [0.005, 0.01],
                    "rfr": [0.001, 0.001],
                    "window": -5,
                },
            )

    def test_config_invalid_confidence_level(self):
        """Test config with invalid confidence level."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            AssessmentRequest(
                assessment_name="VaR",
                assessment_type="summary",
                config={
                    "returns": [0.01, 0.02],
                    "bmk": [0.005, 0.01],
                    "rfr": [0.001, 0.001],
                    "confidence_level": 1.5,
                },
            )

    def test_config_with_valid_optional_params(self):
        """Test config with valid optional parameters."""
        req = AssessmentRequest(
            assessment_name="Beta",
            assessment_type="rolling",
            config={
                "returns": [0.01, 0.02, 0.03],
                "bmk": [0.005, 0.01, 0.015],
                "rfr": [0.001, 0.001, 0.001],
                "window": 10,
                "min_periods": 5,
                "ann_factor": 252,
            },
        )
        assert req.config["window"] == 10
        assert req.config["min_periods"] == 5
        assert req.config["ann_factor"] == 252


class TestRunEndpoint:
    @patch("src.app.api.task_queue")
    def test_enqueue_assessment(self, mock_queue, client, mock_job):
        """Test run endpoint enqueues assessment."""
        mock_queue.enqueue.return_value = mock_job

        response = client.post(
            "/run",
            json={
                "assessment_name": "Beta",
                "assessment_type": "summary",
                "config": {
                    "returns": [0.01, 0.02, 0.03],
                    "bmk": [0.005, 0.01, 0.015],
                    "rfr": [0.001, 0.001, 0.001],
                },
            },
        )

        assert response.status_code == 200
        assert response.json() == {"job_id": "test_job_123"}
        mock_queue.enqueue.assert_called_once()

    def test_invalid_assessment_name(self, client):
        """Test run endpoint with invalid assessment name."""
        response = client.post(
            "/run",
            json={
                "assessment_name": "InvalidName",
                "assessment_type": "summary",
                "config": {
                    "returns": [0.01, 0.02],
                    "bmk": [0.005, 0.01],
                    "rfr": [0.001, 0.001],
                },
            },
        )

        assert response.status_code == 422  # Validation error

    def test_missing_config(self, client):
        """Test run endpoint with missing config."""
        response = client.post(
            "/run",
            json={
                "assessment_name": "Beta",
                "assessment_type": "summary",
                "config": {"returns": [0.01, 0.02]},
            },
        )

        assert response.status_code == 422  # Validation error


class TestSerializeResult:
    def test_serialize_none(self):
        """Test serializing None."""
        assert serialize_result(None) is None

    def test_serialize_dict_with_series(self):
        """Test serializing dict with pandas Series."""
        result = {
            "summary": pd.Series([1.0, 2.0, np.nan]),
            "name": "Beta",
        }
        serialized = serialize_result(result)
        assert serialized["summary"] == [1.0, 2.0, None]
        assert serialized["name"] == "Beta"

    def test_serialize_dict_with_numpy_types(self):
        """Test serializing dict with numpy types."""
        result = {
            "value": np.float64(1.5),
            "count": np.int64(10),
            "nan_value": np.float64(np.nan),
        }
        serialized = serialize_result(result)
        assert serialized["value"] == 1.5
        assert isinstance(serialized["value"], float)
        assert serialized["count"] == 10
        assert isinstance(serialized["count"], int)
        assert serialized["nan_value"] is None

    def test_serialize_series_directly(self):
        """Test serializing pandas Series directly."""
        series = pd.Series([1.0, 2.0, np.nan])
        serialized = serialize_result(series)
        assert serialized == [1.0, 2.0, None]

    def test_serialize_numpy_scalar(self):
        """Test serializing numpy scalar."""
        assert serialize_result(np.float64(1.5)) == 1.5
        assert serialize_result(np.int64(10)) == 10
        assert serialize_result(np.float64(np.nan)) is None

    def test_serialize_dict_with_enum(self):
        """Test serializing dict with enum."""
        from src.constants import AssessmentName

        result = {"name": AssessmentName.Beta}
        serialized = serialize_result(result)
        assert serialized["name"] == "Beta"

    def test_serialize_regular_types(self):
        """Test serializing regular Python types."""
        result = {"string": "test", "number": 42, "float": 1.5}
        serialized = serialize_result(result)
        assert serialized == result

    def test_serialize_nested_dict(self):
        """Test serializing nested dict with mixed types."""
        result = {
            "summary": {"value": np.float64(1.5)},
            "rolling": pd.Series([0.1, 0.2, np.nan]),
        }
        serialized = serialize_result(result)
        assert serialized["summary"]["value"] == 1.5
        assert serialized["rolling"] == [0.1, 0.2, None]
