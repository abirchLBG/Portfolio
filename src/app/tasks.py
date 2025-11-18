# tasks.py
import pandas as pd
from typing import Any, Dict

from src.constants import AssessmentName
from src.dataclasses.assessment_config import AssessmentConfig
from src.evaluation import ALL_ASSESSMENTS


def add_numbers(a: int, b: int):
    return a + b


def run_assessment(
    assessment_name: str,
    assessment_type: str,
    config_dict: Dict[str, Any],
):
    """
    Run an assessment with the given configuration.

    Args:
        assessment_name: Name of the assessment (e.g., "beta", "sharpe_ratio")
        assessment_type: Type of assessment (e.g., "summary", "rolling", "expanding")
        config_dict: Dictionary containing returns, bmk, rfr as lists, plus optional params

    Returns:
        Dictionary with assessment results including name, type, result, and time
    """
    # Convert lists back to pandas Series
    config_dict = config_dict.copy()

    # Extract optional series names
    returns_name = config_dict.pop("returns_name", None)
    bmk_name = config_dict.pop("bmk_name", None)
    rfr_name = config_dict.pop("rfr_name", None)

    if "returns" in config_dict and isinstance(config_dict["returns"], list):
        config_dict["returns"] = pd.Series(config_dict["returns"], name=returns_name)
    if "bmk" in config_dict and isinstance(config_dict["bmk"], list):
        config_dict["bmk"] = pd.Series(config_dict["bmk"], name=bmk_name)
    if "rfr" in config_dict and isinstance(config_dict["rfr"], list):
        config_dict["rfr"] = pd.Series(config_dict["rfr"], name=rfr_name)

    # Get the assessment class from the registry
    try:
        assessment_enum = AssessmentName[assessment_name]
        assessment_class = ALL_ASSESSMENTS[assessment_enum]
    except KeyError:
        raise ValueError(
            f"Unknown assessment: {assessment_name}. Available: {list(ALL_ASSESSMENTS.keys())}"
        )

    # Create config and assessment instance
    config = AssessmentConfig(**config_dict)
    assessment = assessment_class(config=config)

    # Run the assessment
    output = assessment._run(assessment_type)

    return output
