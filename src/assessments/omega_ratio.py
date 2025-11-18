from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from src.assessments.base_assessment import BaseAssessment
from src.constants import AssessmentName


@dataclass(kw_only=True)
class OmegaRatio(BaseAssessment):
    """Omega Ratio Assessment

    Formula:
        Omega = Sum(Returns > Threshold) / Sum(Threshold - Returns < Threshold)

    Description:
        Probability-weighted ratio of gains versus losses relative to a threshold.
        Higher values indicate better risk-adjusted performance.
    """

    name: ClassVar[AssessmentName] = AssessmentName.OmegaRatio

    @staticmethod
    def _summary(
        returns: pd.Series, threshold: float = 0.0, ann_factor: int = 252, **kwargs
    ) -> float:
        excess = returns - threshold
        gains = excess[excess > 0].sum()
        losses = -excess[excess < 0].sum()

        return float(gains / losses) if losses > 0 else np.inf

    @staticmethod
    def _rolling(
        returns: pd.Series,
        threshold: float = 0.0,
        window: int = 252,
        ann_factor: int = 252,
        **kwargs,
    ) -> pd.Series:
        def calc_omega(x):
            excess = x - threshold
            gains = excess[excess > 0].sum()
            losses = -excess[excess < 0].sum()
            return float(gains / losses) if losses > 0 else np.inf

        return returns.rolling(window=window).apply(calc_omega, raw=False)

    @staticmethod
    def _expanding(
        returns: pd.Series,
        threshold: float = 0.0,
        min_periods: int = 21,
        ann_factor: int = 252,
        **kwargs,
    ) -> pd.Series:
        def calc_omega(x):
            excess = x - threshold
            gains = excess[excess > 0].sum()
            losses = -excess[excess < 0].sum()
            return float(gains / losses) if losses > 0 else np.inf

        return returns.expanding(min_periods=min_periods).apply(calc_omega, raw=False)
