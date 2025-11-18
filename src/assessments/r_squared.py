from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from src.assessments.base_assessment import BaseAssessment
from src.constants import AssessmentName


@dataclass(kw_only=True)
class RSquared(BaseAssessment):
    """R-Squared Assessment

    Formula:
        R² = (Correlation(R_p, R_bmk))²

    Description:
        Coefficient of determination measuring how much of the portfolio's movements
        can be explained by benchmark movements. Values range from 0 to 1.
        - 1.0: Perfect correlation with benchmark
        - 0.0: No correlation with benchmark
    """

    name: ClassVar[AssessmentName] = AssessmentName.RSquared

    @staticmethod
    def _summary(returns: pd.Series, bmk: pd.Series, **kwargs) -> float:
        correlation = np.corrcoef(returns, bmk)[0, 1]
        return float(correlation**2)

    @staticmethod
    def _rolling(
        returns: pd.Series, bmk: pd.Series, window: int, **kwargs
    ) -> pd.Series:
        correlation = returns.rolling(window=window).corr(bmk)
        return correlation**2

    @staticmethod
    def _expanding(
        returns: pd.Series, bmk: pd.Series, min_periods: int = 21, **kwargs
    ) -> pd.Series:
        correlation = returns.expanding(min_periods=min_periods).corr(bmk)
        return correlation**2
