from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.assessments.base_assessment import BaseAssessment


@dataclass(kw_only=True)
class TrackingError(BaseAssessment):
    @staticmethod
    def _summary(returns: pd.Series, bmk: pd.Series, ann_factor: int = 252) -> float:
        return float((returns - bmk).std() * np.sqrt(ann_factor))

    @staticmethod
    def _rolling(
        returns: pd.Series, bmk: pd.Series, window: int = 252, ann_factor: int = 252
    ) -> pd.Series:
        return (returns - bmk).rolling(window).std() * np.sqrt(ann_factor)

    @staticmethod
    def _expanding(
        returns: pd.Series, bmk: pd.Series, min_periods: int = 21, ann_factor: int = 252
    ) -> pd.Series:
        return (returns - bmk).expanding(min_periods).std() * np.sqrt(ann_factor)

    def summary(self) -> float:
        return TrackingError._summary(
            returns=self.config.returns,
            bmk=self.config.bmk,
            ann_factor=self.config.ann_factor,
        )

    def rolling(self) -> pd.Series:
        return TrackingError._rolling(
            returns=self.config.returns,
            bmk=self.config.bmk,
            window=self.config.window,
            ann_factor=self.config.ann_factor,
        )

    def expanding(self) -> pd.Series:
        return TrackingError._expanding(
            returns=self.config.returns,
            bmk=self.config.bmk,
            min_periods=self.config.expanding_min_periods,
            ann_factor=self.config.ann_factor,
        )
