from dataclasses import dataclass
from typing import ClassVar

import pandas as pd

from src.assessments.base_assessment import BaseAssessment
from src.constants import AssessmentName


@dataclass(kw_only=True)
class MeanReturn(BaseAssessment):
    """Mean Return Assessment

    Formula:
        Mean Return = E[R]
        Annualized = Mean * ann_factor (arithmetic)

    Description:
        Average return over the period. Can be expressed as simple average
        or annualized. This is the expected return of the portfolio.
    """

    name: ClassVar[AssessmentName] = AssessmentName.MeanReturn

    @staticmethod
    def _summary(returns: pd.Series, ann_factor: int = 252, **kwargs) -> float:
        return float(returns.mean() * ann_factor)

    @staticmethod
    def _rolling(
        returns: pd.Series, window: int, ann_factor: int = 252, **kwargs
    ) -> pd.Series:
        return returns.rolling(window=window).mean() * ann_factor

    @staticmethod
    def _expanding(
        returns: pd.Series, min_periods: int = 21, ann_factor: int = 252, **kwargs
    ) -> pd.Series:
        return returns.expanding(min_periods=min_periods).mean() * ann_factor
