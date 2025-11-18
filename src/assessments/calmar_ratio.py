from dataclasses import dataclass
from typing import ClassVar

import pandas as pd


from src.assessments.base_assessment import BaseAssessment
from src.assessments.max_drawdown import MaxDrawdown
from src.assessments.cagr import CAGR
from src.constants import AssessmentName


@dataclass(kw_only=True)
class CalmarRatio(BaseAssessment):
    name: ClassVar[AssessmentName] = AssessmentName.CalmarRatio

    @staticmethod
    def _summary(returns: pd.Series, ann_factor: int = 252, **kwargs) -> float:
        cagr: float = CAGR._summary(returns=returns, ann_factor=ann_factor)
        max_dd: float = MaxDrawdown._summary(returns=returns)

        return cagr / abs(max_dd)

    @staticmethod
    def _rolling(
        returns: pd.Series, window: int, ann_factor: int = 252, **kwargs
    ) -> pd.Series:
        rolling_cagr: pd.Series = CAGR._rolling(
            returns=returns, window=window, ann_factor=ann_factor
        )
        rolling_max_dd: pd.Series = MaxDrawdown._rolling(returns=returns, window=window)

        return rolling_cagr / rolling_max_dd.abs()

    @staticmethod
    def _expanding(
        returns: pd.Series, min_periods: int, ann_factor: int = 252, **kwargs
    ) -> pd.Series:
        expanding_cagr: pd.Series = CAGR._expanding(
            returns=returns, min_periods=min_periods, ann_factor=ann_factor
        )
        expanding_max_dd: pd.Series = MaxDrawdown._expanding(
            returns=returns, min_periods=min_periods
        )

        return expanding_cagr / expanding_max_dd.abs()
