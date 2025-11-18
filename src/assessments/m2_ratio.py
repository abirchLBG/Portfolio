from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from src.assessments.base_assessment import BaseAssessment
from src.constants import AssessmentName


@dataclass(kw_only=True)
class M2Ratio(BaseAssessment):
    """Modigliani-Modigliani (M²) Ratio Assessment

    Formula:
        M² = (R_p - R_f) * (σ_bmk / σ_p) + R_f - R_bmk

    Description:
        Risk-adjusted performance measure that represents the return a portfolio
        would have earned if it had the same risk (volatility) as the benchmark.
        Expressed in percentage terms, making it more intuitive than Sharpe ratio.
    """

    name: ClassVar[AssessmentName] = AssessmentName.M2Ratio

    @staticmethod
    def _summary(
        returns: pd.Series,
        bmk: pd.Series,
        rfr: pd.Series,
        ann_factor: int = 252,
        **kwargs,
    ) -> float:
        # Annualized returns and volatilities
        portfolio_ret = returns.mean() * ann_factor
        bmk_ret = bmk.mean() * ann_factor
        rfr_mean = rfr.mean() * ann_factor

        portfolio_vol = returns.std() * np.sqrt(ann_factor)
        bmk_vol = bmk.std() * np.sqrt(ann_factor)

        if portfolio_vol == 0:
            return np.nan

        # M² = (R_p - R_f) * (σ_bmk / σ_p) + R_f - R_bmk
        m2 = (portfolio_ret - rfr_mean) * (bmk_vol / portfolio_vol) + rfr_mean - bmk_ret

        return float(m2)

    @staticmethod
    def _rolling(
        returns: pd.Series,
        bmk: pd.Series,
        rfr: pd.Series,
        window: int,
        ann_factor: int = 252,
        **kwargs,
    ) -> pd.Series:
        def calc_m2(idx):
            if idx < window:
                return np.nan

            ret_window = returns.iloc[idx - window : idx]
            bmk_window = bmk.iloc[idx - window : idx]
            rfr_window = rfr.iloc[idx - window : idx]

            portfolio_ret = ret_window.mean() * ann_factor
            bmk_ret = bmk_window.mean() * ann_factor
            rfr_mean = rfr_window.mean() * ann_factor

            portfolio_vol = ret_window.std() * np.sqrt(ann_factor)
            bmk_vol = bmk_window.std() * np.sqrt(ann_factor)

            if portfolio_vol == 0:
                return np.nan

            return (
                (portfolio_ret - rfr_mean) * (bmk_vol / portfolio_vol)
                + rfr_mean
                - bmk_ret
            )

        return pd.Series([calc_m2(i) for i in range(len(returns))], index=returns.index)

    @staticmethod
    def _expanding(
        returns: pd.Series,
        bmk: pd.Series,
        rfr: pd.Series,
        min_periods: int = 21,
        ann_factor: int = 252,
        **kwargs,
    ) -> pd.Series:
        def calc_m2(idx):
            if idx < min_periods - 1:
                return np.nan

            ret_window = returns.iloc[: idx + 1]
            bmk_window = bmk.iloc[: idx + 1]
            rfr_window = rfr.iloc[: idx + 1]

            portfolio_ret = ret_window.mean() * ann_factor
            bmk_ret = bmk_window.mean() * ann_factor
            rfr_mean = rfr_window.mean() * ann_factor

            portfolio_vol = ret_window.std() * np.sqrt(ann_factor)
            bmk_vol = bmk_window.std() * np.sqrt(ann_factor)

            if portfolio_vol == 0:
                return np.nan

            return (
                (portfolio_ret - rfr_mean) * (bmk_vol / portfolio_vol)
                + rfr_mean
                - bmk_ret
            )

        return pd.Series([calc_m2(i) for i in range(len(returns))], index=returns.index)
