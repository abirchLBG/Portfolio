from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from src.assessments.base_assessment import BaseAssessment
from src.constants import AssessmentName


@dataclass(kw_only=True)
class SortinoRatio(BaseAssessment):
    name: ClassVar[AssessmentName] = AssessmentName.SortinoRatio

    @staticmethod
    def _summary(
        returns: pd.Series,
        target: float = 0.0,
        ann_factor: int = 252,
        rfr: pd.Series | None = None,
        **kwargs,
    ) -> float:
        # Use rfr if provided, otherwise use target
        if rfr is not None:
            excess: pd.Series = returns - rfr
        else:
            excess: pd.Series = returns - target

        excess_downside: pd.Series = excess.where(excess < 0, 0.0)
        excess_downside_deviation: float = np.sqrt(np.mean(np.square(excess_downside)))

        if excess_downside_deviation > 0:
            return float(
                excess.mean() / excess_downside_deviation * np.sqrt(ann_factor)
            )
        else:
            # No downside risk - return inf if positive excess, -inf if negative
            mean_excess = excess.mean()
            return float(np.inf) if mean_excess > 0 else float(-np.inf)

    @staticmethod
    def _rolling(
        returns: pd.Series,
        window: int = 252,
        target: float = 0.0,
        ann_factor: int = 252,
        rfr: pd.Series | None = None,
        **kwargs,
    ) -> pd.Series:
        # Use rfr if provided, otherwise use target
        if rfr is not None:
            excess: pd.Series = returns - rfr
        else:
            excess: pd.Series = returns - target

        excess_downside: pd.Series = excess.where(excess < 0, 0.0)

        rolling_excess_mean: pd.Series = excess.rolling(window).mean()
        rolling_excess_downside_deviation: pd.Series = (
            excess_downside.pow(2).rolling(window).mean().pow(0.5)
        )

        return (
            rolling_excess_mean
            / rolling_excess_downside_deviation
            * np.sqrt(ann_factor)
        )

    @staticmethod
    def _expanding(
        returns: pd.Series,
        min_periods: int = 252,
        target: float = 0.0,
        ann_factor: int = 252,
        rfr: pd.Series | None = None,
        **kwargs,
    ) -> pd.Series:
        # Use rfr if provided, otherwise use target
        if rfr is not None:
            excess: pd.Series = returns - rfr
        else:
            excess: pd.Series = returns - target

        excess_downside: pd.Series = excess.where(excess < 0, 0.0)

        rolling_excess_mean: pd.Series = excess.expanding(min_periods).mean()
        rolling_excess_downside_deviation: pd.Series = (
            excess_downside.pow(2).expanding(min_periods).mean().pow(0.5)
        )

        return (
            rolling_excess_mean
            / rolling_excess_downside_deviation
            * np.sqrt(ann_factor)
        )
