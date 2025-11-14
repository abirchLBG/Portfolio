from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.assessments.base_assessment import BaseAssessment


@dataclass(kw_only=True)
class Beta(BaseAssessment):
    """Beta Assessment

    Formula:
        Beta_p = Cov(R_p, R_bmk) / Var(R_bmk)

    Description:
        Beta is a measure of a portfolio's sensitivity to market movements.
    """

    @staticmethod
    def _summary(returns: pd.Series, bmk: pd.Series) -> float:
        cov: np.ndarray = np.cov(returns, bmk)

        return float(cov[0, 1] / cov[1, 1])

    @staticmethod
    def _rolling(returns: pd.Series, bmk: pd.Series, window: int) -> pd.Series:
        rolling_cov: pd.Series = returns.rolling(window).cov(bmk)
        rolling_var: pd.Series = bmk.rolling(window).var()

        return rolling_cov / rolling_var

    @staticmethod
    def _expanding(
        returns: pd.Series, bmk: pd.Series, min_periods: int = 21
    ) -> pd.Series:
        expanding_cov: pd.Series = returns.expanding(min_periods).cov(bmk)
        expanding_var: pd.Series = bmk.expanding(min_periods).var()

        return expanding_cov / expanding_var

    def summary(self) -> float:
        return self._summary(returns=self.config.returns, bmk=self.config.bmk)

    def rolling(self) -> pd.Series:
        return self._rolling(
            returns=self.config.returns, bmk=self.config.bmk, window=self.config.window
        )

    def expanding(self) -> pd.Series:
        return self._expanding(
            returns=self.config.returns,
            bmk=self.config.bmk,
            min_periods=self.config.expanding_min_periods,
        )
