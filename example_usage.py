"""
Example usage of the refactored assessment API and RQExecutor.

This script demonstrates:
1. How to use the Evaluation class with different executors
2. How assessments are submitted to the remote API via RQExecutor
"""

import numpy as np
import pandas as pd

from src.constants import AssessmentName
from src.dataclasses.assessment_config import AssessmentConfig
from src.dataclasses.assessment_results import AssessmentType
from src.evaluation import Evaluation, ExecutorType

# Generate sample data
dates = pd.date_range("2020-01-01", periods=252, freq="D")
returns = pd.Series(np.random.randn(252) * 0.01, index=dates)
bmk = pd.Series(np.random.randn(252) * 0.01, index=dates)
rfr = pd.Series(np.ones(252) * 0.0001, index=dates)

# Create assessment configuration
config = AssessmentConfig(returns=returns, bmk=bmk, rfr=rfr)

print("=" * 60)
print("Example 1: Local execution with DummyExecutor")
print("=" * 60)

eval_local = (
    Evaluation(config=config)
    .with_assessments([AssessmentName.Beta, AssessmentName.SharpeRatio])
    .with_assessment_types([AssessmentType.Summary])
    .with_executor(ExecutorType.DEFAULT())
    .run()
)

print("Results:", eval_local._results)
print("\nTimer:")
print(eval_local.timer())

print("\n" + "=" * 60)
print("Example 2: Remote execution with RQExecutor")
print("=" * 60)
print("NOTE: This requires the API to be running (make run)")
print("URL: http://localhost:8000")
print("\nUsage:")
print("""
eval_remote = (
    Evaluation(config=config)
    .with_assessments([AssessmentName.Beta])
    .with_assessment_types([AssessmentType.Summary])
    .with_executor(RQExecutor(api_url="http://localhost:8000"))
    .run()
)

print("Results:", eval_remote._results)
print("Timer:", eval_remote.timer())
""")

print("\n" + "=" * 60)
print("Example 3: Direct API usage (via curl)")
print("=" * 60)
print("""
# Submit an assessment job:
curl -X POST http://localhost:8000/assessment \\
  -H "Content-Type: application/json" \\
  -d '{
    "assessment_name": "Beta",
    "assessment_type": "summary",
    "config": {
      "returns": [0.01, 0.02, -0.01, 0.015],
      "bmk": [0.008, 0.018, -0.012, 0.014],
      "rfr": [0.0001, 0.0001, 0.0001, 0.0001]
    }
  }'

# Check job status (replace JOB_ID with the returned job_id):
curl http://localhost:8000/status/JOB_ID
""")
