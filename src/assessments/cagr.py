from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.assessments.base_assessment import BaseAssessment


@dataclass(kw_only=True)
class CAGR(BaseAssessment):
    @staticmethod
    def _summary(returns: pd.Series, ann_factor: int = 252) -> float:
        return float(np.prod(returns + 1) ** (ann_factor / len(returns)) - 1)

    @staticmethod
    def _rolling(
        returns: pd.Series, window: int = 252, ann_factor: int = 252
    ) -> pd.Series:
        return returns.rolling(window=window).apply(
            CAGR._summary, args=(ann_factor,), raw=True
        )

    @staticmethod
    def _expanding(
        returns: pd.Series, min_periods: int = 21, ann_factor: int = 252
    ) -> pd.Series:
        return returns.expanding(min_periods).apply(
            CAGR._summary,
            args=(ann_factor,),
            raw=True,
        )

    def summary(self) -> float:
        return self._summary(
            returns=self.config.returns, ann_factor=self.config.ann_factor
        )

    def rolling(self) -> pd.Series:
        return self._rolling(
            returns=self.config.returns,
            window=self.config.window,
            ann_factor=self.config.ann_factor,
        )

    def expanding(self) -> pd.Series:
        return self._expanding(
            returns=self.config.returns,
            min_periods=self.config.expanding_min_periods,
            ann_factor=self.config.ann_factor,
        )
