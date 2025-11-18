from dataclasses import dataclass
from typing import ClassVar

import pandas as pd

from src.assessments.base_assessment import BaseAssessment
from src.constants import AssessmentName


@dataclass(kw_only=True)
class CumulativeReturns(BaseAssessment):
    """Cumulative Returns Assessment

    Formula:
        Cumulative Return = (1 + r1) * (1 + r2) * ... * (1 + rn) - 1

    Description:
        Total return over the entire period, accounting for compounding.
        Shows the overall percentage gain or loss.
    """

    name: ClassVar[AssessmentName] = AssessmentName.CumulativeReturns

    @staticmethod
    def _summary(returns: pd.Series, **kwargs) -> float:
        return float((1 + returns).prod() - 1)

    @staticmethod
    def _rolling(returns: pd.Series, window: int, **kwargs) -> pd.Series:
        return returns.rolling(window=window).apply(
            lambda x: (1 + x).prod() - 1, raw=False
        )

    @staticmethod
    def _expanding(returns: pd.Series, min_periods: int = 21, **kwargs) -> pd.Series:
        return returns.expanding(min_periods=min_periods).apply(
            lambda x: (1 + x).prod() - 1, raw=False
        )
