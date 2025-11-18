from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from src.assessments.base_assessment import BaseAssessment
from src.constants import AssessmentName


@dataclass(kw_only=True)
class SemiVariance(BaseAssessment):
    """Semi-Variance Assessment

    Formula:
        Semi-Variance = E[min(R - target, 0)²]

    Description:
        Measures downside volatility only, focusing on returns below a threshold.
        Used in Sortino Ratio calculation. Lower values indicate less downside risk.
    """

    name: ClassVar[AssessmentName] = AssessmentName.SemiVariance

    @staticmethod
    def _summary(
        returns: pd.Series, target: float = 0.0, ann_factor: int = 252, **kwargs
    ) -> float:
        # Calculate deviations below target
        downside = returns - target
        downside[downside > 0] = 0

        # Annualize the semi-variance
        return float(downside.var() * ann_factor)

    @staticmethod
    def _rolling(
        returns: pd.Series,
        target: float = 0.0,
        window: int = 252,
        ann_factor: int = 252,
        **kwargs,
    ) -> pd.Series:
        def calc_semi_var(x):
            downside = x - target
            downside = downside[downside < 0]
            return float(downside.var() * ann_factor) if len(downside) > 0 else np.nan

        return returns.rolling(window=window).apply(calc_semi_var, raw=False)

    @staticmethod
    def _expanding(
        returns: pd.Series,
        target: float = 0.0,
        min_periods: int = 21,
        ann_factor: int = 252,
        **kwargs,
    ) -> pd.Series:
        def calc_semi_var(x):
            downside = x - target
            downside = downside[downside < 0]
            return float(downside.var() * ann_factor) if len(downside) > 0 else np.nan

        return returns.expanding(min_periods=min_periods).apply(
            calc_semi_var, raw=False
        )
