from fastapi import FastAPI
from rq.job import Job
from src.app.task_queue import task_queue
from src.app.tasks import add_numbers, run_assessment

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any

from src.constants import AssessmentName
from src.dataclasses.assessment_results import AssessmentType

app = FastAPI()


class AssessmentRequest(BaseModel):
    assessment_name: str = Field(
        ...,
        description="Name of the assessment to run (must match AssessmentName enum)",
    )
    assessment_type: str = Field(
        ..., description="Type of assessment: 'summary', 'rolling', or 'expanding'"
    )
    config: Dict[str, Any] = Field(
        ...,
        description="Configuration dict containing returns, bmk, rfr (as lists), and optional params",
    )

    @field_validator("assessment_name")
    @classmethod
    def validate_assessment_name(cls, v: str) -> str:
        """Validate that assessment_name is valid."""
        try:
            # Try to convert to AssessmentName enum
            AssessmentName[v]
            return v
        except KeyError:
            valid_names = [name.name for name in AssessmentName]
            raise ValueError(
                f"Invalid assessment_name: '{v}'. Must be one of: {valid_names}"
            )

    @field_validator("assessment_type")
    @classmethod
    def validate_assessment_type(cls, v: str) -> str:
        """Validate that assessment_type is valid."""
        valid_types = [t.value for t in AssessmentType]
        if v not in valid_types:
            raise ValueError(
                f"Invalid assessment_type: '{v}'. Must be one of: {valid_types}"
            )
        return v

    @field_validator("config")
    @classmethod
    def validate_config(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that config contains required fields."""
        required_fields = ["returns", "bmk", "rfr"]
        missing = [f for f in required_fields if f not in v]

        if missing:
            raise ValueError(
                f"Config missing required fields: {missing}. "
                f"Required: {required_fields}"
            )

        # Validate that returns, bmk, rfr are lists (will be converted to Series)
        for field in required_fields:
            if not isinstance(v[field], list):
                raise ValueError(
                    f"Config field '{field}' must be a list, got {type(v[field]).__name__}"
                )

            if len(v[field]) == 0:
                raise ValueError(f"Config field '{field}' cannot be empty")

        # Validate optional numeric parameters if present
        numeric_params = ["ann_factor", "window", "min_periods", "confidence_level"]
        for param in numeric_params:
            if param in v:
                if not isinstance(v[param], (int, float)):
                    raise ValueError(
                        f"Config parameter '{param}' must be numeric, got {type(v[param]).__name__}"
                    )
                if param in ["window", "min_periods", "ann_factor"] and v[param] <= 0:
                    raise ValueError(f"Config parameter '{param}' must be positive")
                if param == "confidence_level" and not (0 < v[param] < 1):
                    raise ValueError(
                        f"Config parameter 'confidence_level' must be between 0 and 1, got {v[param]}"
                    )

        return v


@app.get("/")
def ping():
    return "pong"


@app.post("/add/")
def enqueue_add(a: int, b: int):
    job = task_queue.enqueue(add_numbers, a, b)
    return {"job_id": job.get_id()}


def serialize_result(result: Any) -> Any:
    """Convert assessment result to JSON-serializable format."""
    if result is None:
        return None

    if isinstance(result, dict):
        # Handle dict results (from assessment._run())
        serialized = {}
        for key, value in result.items():
            if isinstance(value, pd.Series):
                # Convert Series to list, replacing NaN with None
                serialized[key] = value.replace({np.nan: None}).tolist()
            elif isinstance(value, (np.integer, np.floating)):
                # Convert numpy types to Python types
                if np.isnan(value):
                    serialized[key] = None
                else:
                    serialized[key] = (
                        float(value) if isinstance(value, np.floating) else int(value)
                    )
            elif hasattr(value, "value"):
                # Handle enums (like AssessmentName)
                serialized[key] = value.value
            else:
                serialized[key] = value
        return serialized
    elif isinstance(result, pd.Series):
        # Convert Series directly
        return result.replace({np.nan: None}).tolist()
    elif isinstance(result, (np.integer, np.floating)):
        if np.isnan(result):
            return None
        return float(result) if isinstance(result, np.floating) else int(result)
    else:
        return result


@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = Job.fetch(job_id, connection=task_queue.connection)
    result = job.result if job.is_finished else None

    # Serialize the result to handle pandas/numpy types
    if result is not None:
        result = serialize_result(result)

    return {
        "job_id": job.id,
        "status": job.get_status(),
        "result": result,
    }


@app.post("/run")
def enqueue_assessment(req: AssessmentRequest):
    job = task_queue.enqueue(
        run_assessment,
        req.assessment_name,
        req.assessment_type,
        req.config,
    )
    return {"job_id": job.id}
