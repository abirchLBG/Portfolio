from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.assessments.base_assessment import BaseAssessment


@dataclass(kw_only=True)
class SharpeRatio(BaseAssessment):
    @staticmethod
    def _summary(
        returns: pd.Series, rfr: pd.Series, ann_factor: int = 252, **kwargs
    ) -> float:
        excess: pd.Series = returns - rfr
        excess_std: float = excess.std()

        return (
            float((excess.mean() * np.sqrt(ann_factor)) / (excess_std))
            if excess_std > 0
            else np.nan
        )

    @staticmethod
    def _rolling(
        returns: pd.Series, rfr: pd.Series, window: int, ann_factor: int = 252, **kwargs
    ) -> pd.Series:
        excess: pd.Series = returns - rfr

        return excess.rolling(window=window).apply(
            lambda x: float(x.mean() * np.sqrt(ann_factor) / x.std())
            if x.std() > 0
            else np.nan,
            raw=False,
        )

    @staticmethod
    def _expanding(
        returns: pd.Series,
        rfr: pd.Series,
        min_periods: int = 21,
        ann_factor: int = 252,
        **kwargs,
    ) -> pd.Series:
        excess: pd.Series = returns - rfr

        return excess.expanding(min_periods=min_periods).apply(
            lambda x: float(x.mean() * np.sqrt(ann_factor) / x.std())
            if x.std() > 0
            else np.nan,
            raw=False,
        )
