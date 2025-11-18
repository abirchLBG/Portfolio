from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from src.assessments.base_assessment import BaseAssessment
from src.assessments.beta import Beta
from src.constants import AssessmentName


@dataclass(kw_only=True)
class TreynorRatio(BaseAssessment):
    name: ClassVar[AssessmentName] = AssessmentName.TreynorRatio

    @staticmethod
    def _summary(
        returns: pd.Series,
        rfr: pd.Series,
        bmk: pd.Series,
        ann_factor: int = 252,
        **kwargs,
    ) -> float:
        beta: float = Beta._summary(returns=returns, bmk=bmk)
        excess: pd.Series = returns - rfr

        return excess.mean() * ann_factor / beta if beta != 0 else np.nan

    @staticmethod
    def _rolling(
        returns: pd.Series,
        rfr: pd.Series,
        bmk: pd.Series,
        window: int = 252,
        ann_factor: int = 252,
        **kwargs,
    ) -> pd.Series:
        rolling_beta: pd.Series = Beta._rolling(returns=returns, bmk=bmk, window=window)
        excess: pd.Series = returns - rfr

        return (excess.rolling(window).mean() * ann_factor / rolling_beta).where(
            rolling_beta != 0, np.nan
        )

    @staticmethod
    def _expanding(
        returns: pd.Series,
        rfr: pd.Series,
        bmk: pd.Series,
        min_periods: int = 21,
        ann_factor: int = 252,
        **kwargs,
    ) -> pd.Series:
        expanding_beta: pd.Series = Beta._expanding(
            returns=returns, bmk=bmk, min_periods=min_periods
        )
        excess: pd.Series = returns - rfr

        return (
            excess.expanding(min_periods).mean() * ann_factor / expanding_beta
        ).where(expanding_beta != 0, np.nan)
