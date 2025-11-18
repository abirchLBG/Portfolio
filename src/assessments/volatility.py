from dataclasses import dataclass
from typing import ClassVar

import pandas as pd

from src.assessments.base_assessment import BaseAssessment
from src.constants import AssessmentName


@dataclass(kw_only=True)
class Volatility(BaseAssessment):
    """Volatility (Standard Deviation) Assessment

    Formula:
        Vol_p = StdDev(R_p) * sqrt(ann_factor)

    Description:
        Annualized standard deviation of returns, measuring total risk.
    """

    name: ClassVar[AssessmentName] = AssessmentName.Volatility

    @staticmethod
    def _summary(returns: pd.Series, ann_factor: int = 252, **kwargs) -> float:
        return float(returns.std() * (ann_factor**0.5))

    @staticmethod
    def _rolling(
        returns: pd.Series, window: int, ann_factor: int = 252, **kwargs
    ) -> pd.Series:
        return returns.rolling(window).std() * (ann_factor**0.5)

    @staticmethod
    def _expanding(
        returns: pd.Series, min_periods: int = 21, ann_factor: int = 252, **kwargs
    ) -> pd.Series:
        return returns.expanding(min_periods).std() * (ann_factor**0.5)
