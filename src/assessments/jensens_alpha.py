from dataclasses import dataclass

import pandas as pd

from src.assessments.base_assessment import BaseAssessment
from src.assessments.beta import Beta


@dataclass(kw_only=True)
class JensensAlpha(BaseAssessment):
    @staticmethod
    def _summary(
        returns: pd.Series,
        rfr: pd.Series,
        bmk: pd.Series,
        ann_factor: int = 252,
        **kwargs,
    ) -> float:
        beta: float = Beta._summary(returns=returns, bmk=bmk)

        return ((returns - rfr) - beta * (bmk - rfr)).mean() * ann_factor

    @staticmethod
    def _rolling(
        returns: pd.Series,
        rfr: pd.Series,
        bmk: pd.Series,
        window: int,
        ann_factor: int = 252,
        **kwargs,
    ) -> pd.Series:
        rolling_beta: pd.Series = Beta._rolling(returns=returns, bmk=bmk, window=window)

        return ((returns - rfr) - rolling_beta * (bmk - rfr)).rolling(
            window
        ).mean() * ann_factor

    @staticmethod
    def _expanding(
        returns: pd.Series,
        rfr: pd.Series,
        bmk: pd.Series,
        min_periods: int,
        ann_factor: int = 252,
        **kwargs,
    ) -> pd.Series:
        rolling_beta: pd.Series = Beta._expanding(
            returns=returns, bmk=bmk, min_periods=min_periods
        )

        return ((returns - rfr) - rolling_beta * (bmk - rfr)).expanding(
            min_periods
        ).mean() * ann_factor
