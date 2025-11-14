from dataclasses import dataclass

import pandas as pd


from src.assessments.base_assessment import BaseAssessment
from src.assessments.max_drawdown import MaxDrawdown
from src.assessments.cagr import CAGR


@dataclass(kw_only=True)
class CalmarRatio(BaseAssessment):
    @staticmethod
    def _summary(returns: pd.Series, ann_factor: int = 252) -> float:
        cagr: float = CAGR._summary(returns=returns, ann_factor=ann_factor)
        max_dd: float = MaxDrawdown._summary(returns=returns)

        return cagr / abs(max_dd)

    @staticmethod
    def _rolling(returns: pd.Series, window: int, ann_factor: int = 252) -> pd.Series:
        rolling_cagr: pd.Series = CAGR._rolling(
            returns=returns, window=window, ann_factor=ann_factor
        )
        rolling_max_dd: pd.Series = MaxDrawdown._rolling(returns=returns, window=window)

        return rolling_cagr / rolling_max_dd.abs()

    @staticmethod
    def _expanding(
        returns: pd.Series, min_periods: int, ann_factor: int = 252
    ) -> pd.Series:
        expanding_cagr: pd.Series = CAGR._expanding(
            returns=returns, min_periods=min_periods, ann_factor=ann_factor
        )
        expanding_max_dd: pd.Series = MaxDrawdown._expanding(
            returns=returns, min_periods=min_periods
        )

        return expanding_cagr / expanding_max_dd.abs()

    def summary(self) -> float:
        return CalmarRatio._summary(
            returns=self.config.returns, ann_factor=self.config.ann_factor
        )

    def rolling(self) -> pd.Series:
        return CalmarRatio._rolling(
            returns=self.config.returns,
            window=self.config.window,
            ann_factor=self.config.ann_factor,
        )

    def expanding(self) -> pd.Series:
        return CalmarRatio._expanding(
            returns=self.config.returns,
            min_periods=self.config.expanding_min_periods,
            ann_factor=self.config.ann_factor,
        )
