from dataclasses import dataclass
from typing import ClassVar

import pandas as pd
import numpy as np

from src.assessments.base_assessment import BaseAssessment
from src.constants import AssessmentName


@dataclass
class MaxDrawdown(BaseAssessment):
    name: ClassVar[AssessmentName] = AssessmentName.MaxDrawdown

    @staticmethod
    def _summary(returns: pd.Series, **kwargs) -> float:
        cum_returns: np.ndarray = np.cumprod(returns + 1)
        running_max: np.ndarray = np.maximum.accumulate(cum_returns)
        drawdown: np.ndarray = cum_returns / running_max - 1
        max_dd: float = float(drawdown.min())

        return max_dd

    @staticmethod
    def _rolling(returns: pd.Series, window: int = 252, **kwargs) -> pd.Series:
        return returns.rolling(window=window).apply(MaxDrawdown._summary, raw=True)

    @staticmethod
    def _expanding(returns: pd.Series, min_periods: int = 21, **kwargs) -> pd.Series:
        return returns.expanding(min_periods=min_periods).apply(
            MaxDrawdown._summary,
            raw=True,
        )
