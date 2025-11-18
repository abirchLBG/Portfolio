from dataclasses import dataclass
from typing import ClassVar

import pandas as pd
from scipy import stats

from src.assessments.base_assessment import BaseAssessment
from src.constants import AssessmentName


@dataclass(kw_only=True)
class Skewness(BaseAssessment):
    """Skewness Assessment

    Formula:
        Skewness = E[(X - μ)³] / σ³

    Description:
        Measures the asymmetry of the return distribution.
        - Negative skew: left tail is longer (more extreme losses)
        - Positive skew: right tail is longer (more extreme gains)
        - Zero: symmetric distribution
    """

    name: ClassVar[AssessmentName] = AssessmentName.Skewness

    @staticmethod
    def _summary(returns: pd.Series, **kwargs) -> float:
        return float(stats.skew(returns, bias=False))

    @staticmethod
    def _rolling(returns: pd.Series, window: int, **kwargs) -> pd.Series:
        return returns.rolling(window=window).apply(
            lambda x: float(stats.skew(x, bias=False)), raw=False
        )

    @staticmethod
    def _expanding(returns: pd.Series, min_periods: int = 21, **kwargs) -> pd.Series:
        return returns.expanding(min_periods=min_periods).apply(
            lambda x: float(stats.skew(x, bias=False)), raw=False
        )
