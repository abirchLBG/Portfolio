from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from src.assessments.base_assessment import BaseAssessment
from src.constants import AssessmentName


@dataclass(kw_only=True)
class UlcerIndex(BaseAssessment):
    """Ulcer Index Assessment

    Formula:
        UI = sqrt(Sum[(Drawdown_i)²] / N)

    Description:
        Measures the depth and duration of drawdowns. Unlike standard deviation,
        it only considers downside volatility and the time spent in drawdowns.
        Lower values indicate less downside risk.

        Named after the "ulcer" one might get from watching portfolio decline.
    """

    name: ClassVar[AssessmentName] = AssessmentName.UlcerIndex

    @staticmethod
    def _summary(returns: pd.Series, **kwargs) -> float:
        # Calculate cumulative returns
        cumulative = (1 + returns).cumprod()

        # Calculate running maximum
        running_max = cumulative.expanding().max()

        # Calculate drawdown percentage from peak
        drawdown = ((cumulative - running_max) / running_max) * 100

        # Ulcer Index is RMS of drawdowns
        ulcer = np.sqrt((drawdown**2).mean())

        return float(ulcer)

    @staticmethod
    def _rolling(returns: pd.Series, window: int, **kwargs) -> pd.Series:
        def calc_ulcer(x):
            cumulative = (1 + x).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = ((cumulative - running_max) / running_max) * 100
            return np.sqrt((drawdown**2).mean())

        return returns.rolling(window=window).apply(calc_ulcer, raw=False)

    @staticmethod
    def _expanding(returns: pd.Series, min_periods: int = 21, **kwargs) -> pd.Series:
        def calc_ulcer(x):
            cumulative = (1 + x).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = ((cumulative - running_max) / running_max) * 100
            return np.sqrt((drawdown**2).mean())

        return returns.expanding(min_periods=min_periods).apply(calc_ulcer, raw=False)
