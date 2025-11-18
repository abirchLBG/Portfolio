from dataclasses import dataclass
from typing import ClassVar

import pandas as pd
from scipy import stats

from src.assessments.base_assessment import BaseAssessment
from src.constants import AssessmentName


@dataclass(kw_only=True)
class Kurtosis(BaseAssessment):
    """Kurtosis Assessment

    Formula:
        Kurtosis = E[(X - μ)⁴] / σ⁴ - 3  (excess kurtosis)

    Description:
        Measures the tail risk or "fat-tailedness" of the return distribution.
        - Positive (leptokurtic): fatter tails, more extreme events
        - Negative (platykurtic): thinner tails, fewer extreme events
        - Zero (mesokurtic): normal distribution-like tails
    """

    name: ClassVar[AssessmentName] = AssessmentName.Kurtosis

    @staticmethod
    def _summary(returns: pd.Series, excess: bool = True, **kwargs) -> float:
        # Fisher=True gives excess kurtosis (normal dist = 0)
        # Fisher=False gives raw kurtosis (normal dist = 3)
        return float(stats.kurtosis(returns, bias=False, fisher=excess))

    @staticmethod
    def _rolling(
        returns: pd.Series, window: int, excess: bool = True, **kwargs
    ) -> pd.Series:
        return returns.rolling(window=window).apply(
            lambda x: float(stats.kurtosis(x, bias=False, fisher=excess)), raw=False
        )

    @staticmethod
    def _expanding(
        returns: pd.Series, min_periods: int = 21, excess: bool = True, **kwargs
    ) -> pd.Series:
        return returns.expanding(min_periods=min_periods).apply(
            lambda x: float(stats.kurtosis(x, bias=False, fisher=excess)), raw=False
        )
