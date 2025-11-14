from dataclasses import dataclass

import pandas as pd

from src.assessments.base_assessment import BaseAssessment
from src.assessments.beta import Beta


@dataclass(kw_only=True)
class JensensAlpha(BaseAssessment):
    # def calc(self) -> float:
    #     beta: float = Beta(config=self.config).calc()

    #     # based on CAPM model
    #     # CAPM (R_p) = R_f + Beta * (R_m - R_f)
    #     # alpha = (Rp - Rf) - beta * (Rm - Rf)

    #     self.value: float = (
    #         (self.config.returns - self.config.rfr)
    #         - beta * (self.config.bmk - self.config.rfr)
    #     ).mean() * 252
    #     return self.value

    @staticmethod
    def _summary(
        returns: pd.Series, rfr: pd.Series, bmk: pd.Series, ann_factor: int = 252
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
    ) -> pd.Series:
        rolling_beta: pd.Series = Beta._expanding(
            returns=returns, bmk=bmk, min_periods=min_periods
        )

        return ((returns - rfr) - rolling_beta * (bmk - rfr)).expanding(
            min_periods
        ).mean() * ann_factor

    def summary(self) -> float:
        return self._summary(
            returns=self.config.returns,
            bmk=self.config.bmk,
            rfr=self.config.rfr,
            ann_factor=self.config.ann_factor,
        )

    def rolling(self) -> pd.Series:
        return self._rolling(
            returns=self.config.returns,
            rfr=self.config.rfr,
            bmk=self.config.bmk,
            window=self.config.window,
            ann_factor=self.config.ann_factor,
        )

    def expanding(self) -> pd.Series:
        return self._expanding(
            returns=self.config.returns,
            rfr=self.config.rfr,
            bmk=self.config.bmk,
            min_periods=self.config.expanding_min_periods,
            ann_factor=self.config.ann_factor,
        )
