from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from src.assessments.base_assessment import BaseAssessment
from src.assessments.tracking_error import TrackingError
from src.constants import AssessmentName


@dataclass(kw_only=True)
class InformationRatio(BaseAssessment):
    name: ClassVar[AssessmentName] = AssessmentName.InformationRatio

    @staticmethod
    def _summary(
        returns: pd.Series, bmk: pd.Series, ann_factor: int = 252, **kwargs
    ) -> float:
        tracking_error: float = TrackingError._summary(
            returns=returns, bmk=bmk, ann_factor=1
        )

        return (
            float(
                ((returns - bmk).mean() * ann_factor)
                / (tracking_error * np.sqrt(ann_factor))
            )
            if tracking_error != 0
            else np.nan
        )

    @staticmethod
    def _rolling(
        returns: pd.Series,
        bmk: pd.Series,
        window: int = 252,
        ann_factor: int = 252,
        **kwargs,
    ) -> pd.Series:
        rolling_te: pd.Series = TrackingError._rolling(
            returns=returns, bmk=bmk, window=window, ann_factor=1
        )

        return ((returns - bmk).rolling(window).mean() * ann_factor) / (
            rolling_te * np.sqrt(ann_factor)
        )

    @staticmethod
    def _expanding(
        returns: pd.Series,
        bmk: pd.Series,
        min_periods: int = 21,
        ann_factor: int = 252,
        **kwargs,
    ) -> pd.Series:
        expanding_te: pd.Series = TrackingError._expanding(
            returns=returns, bmk=bmk, min_periods=min_periods, ann_factor=1
        )

        return ((returns - bmk).expanding(min_periods).mean() * ann_factor) / (
            expanding_te * np.sqrt(ann_factor)
        )
