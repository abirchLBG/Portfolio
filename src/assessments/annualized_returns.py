from dataclasses import dataclass
from typing import ClassVar

import pandas as pd

from src.assessments.base_assessment import BaseAssessment
from src.constants import AssessmentName


@dataclass(kw_only=True)
class AnnualizedReturns(BaseAssessment):
    """Annualized Returns Assessment

    Formula:
        Annualized Return = (1 + Mean Daily Return)^ann_factor - 1
        OR
        Annualized Return = Mean Daily Return * ann_factor (simple)

    Description:
        Converts period returns to an annualized figure for comparison.
        Uses geometric mean by default for accuracy over longer periods.
    """

    name: ClassVar[AssessmentName] = AssessmentName.AnnualizedReturns

    @staticmethod
    def _summary(
        returns: pd.Series, ann_factor: int = 252, geometric: bool = True, **kwargs
    ) -> float:
        if geometric:
            # Geometric mean: (1 + r1) * (1 + r2) * ... - 1
            return float((1 + returns).prod() ** (ann_factor / len(returns)) - 1)
        else:
            # Arithmetic mean
            return float(returns.mean() * ann_factor)

    @staticmethod
    def _rolling(
        returns: pd.Series,
        window: int,
        ann_factor: int = 252,
        geometric: bool = True,
        **kwargs,
    ) -> pd.Series:
        if geometric:

            def calc_ann_ret(x):
                return (1 + x).prod() ** (ann_factor / len(x)) - 1

            return returns.rolling(window=window).apply(calc_ann_ret, raw=False)
        else:
            return returns.rolling(window=window).mean() * ann_factor

    @staticmethod
    def _expanding(
        returns: pd.Series,
        min_periods: int = 21,
        ann_factor: int = 252,
        geometric: bool = True,
        **kwargs,
    ) -> pd.Series:
        if geometric:

            def calc_ann_ret(x):
                return (1 + x).prod() ** (ann_factor / len(x)) - 1

            return returns.expanding(min_periods=min_periods).apply(
                calc_ann_ret, raw=False
            )
        else:
            return returns.expanding(min_periods=min_periods).mean() * ann_factor
