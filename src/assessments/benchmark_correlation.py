from dataclasses import dataclass
from typing import ClassVar

import pandas as pd

from src.assessments.base_assessment import BaseAssessment
from src.constants import AssessmentName


@dataclass(kw_only=True)
class BenchmarkCorrelation(BaseAssessment):
    """Benchmark correlation Assessment

    Formula:
        Corr(R_p, R_bmk) = Cov(R_p, R_bmk) / (StdDev(R_p) * StdDev(R_bmk))

    Description:
        Pearson correlation coefficient between portfolio and benchmark returns.
        Measures the strength and direction of the linear relationship.
        Range: -1 (perfect negative) to +1 (perfect positive)
    """

    name: ClassVar[AssessmentName] = AssessmentName.BenchmarkCorrelation

    @staticmethod
    def _summary(returns: pd.Series, bmk: pd.Series, **kwargs) -> float:
        return float(returns.corr(bmk))

    @staticmethod
    def _rolling(
        returns: pd.Series, bmk: pd.Series, window: int, **kwargs
    ) -> pd.Series:
        return returns.rolling(window).corr(bmk)

    @staticmethod
    def _expanding(
        returns: pd.Series, bmk: pd.Series, min_periods: int = 21, **kwargs
    ) -> pd.Series:
        return returns.expanding(min_periods).corr(bmk)
