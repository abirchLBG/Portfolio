from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from src.assessments.base_assessment import BaseAssessment
from src.constants import AssessmentName


@dataclass(kw_only=True)
class TrackingError(BaseAssessment):
    name: ClassVar[AssessmentName] = AssessmentName.TrackingError

    @staticmethod
    def _summary(
        returns: pd.Series, bmk: pd.Series, ann_factor: int = 252, **kwargs
    ) -> float:
        return float((returns - bmk).std() * np.sqrt(ann_factor))

    @staticmethod
    def _rolling(
        returns: pd.Series,
        bmk: pd.Series,
        window: int = 252,
        ann_factor: int = 252,
        **kwargs,
    ) -> pd.Series:
        return (returns - bmk).rolling(window).std() * np.sqrt(ann_factor)

    @staticmethod
    def _expanding(
        returns: pd.Series,
        bmk: pd.Series,
        min_periods: int = 21,
        ann_factor: int = 252,
        **kwargs,
    ) -> pd.Series:
        return (returns - bmk).expanding(min_periods).std() * np.sqrt(ann_factor)
