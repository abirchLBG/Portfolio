from dataclasses import dataclass
from typing import ClassVar

import pandas as pd

from src.assessments.base_assessment import BaseAssessment
from src.constants import AssessmentName


@dataclass(kw_only=True)
class VaR(BaseAssessment):
    """Value at Risk (VaR) Assessment

    Formula:
        VaR_α = Percentile(R_p, α)

    Description:
        Maximum expected loss at a given confidence level (default 95%).
        Expressed as a negative number representing the loss.
        For example, 95% VaR = -0.02 means there's a 5% chance of losing more than 2%.
    """

    name: ClassVar[AssessmentName] = AssessmentName.VaR

    @staticmethod
    def _summary(returns: pd.Series, confidence_level: float = 0.95, **kwargs) -> float:
        """Calculate VaR using historical simulation method."""
        return float(returns.quantile(1 - confidence_level))

    @staticmethod
    def _rolling(
        returns: pd.Series, window: int, confidence_level: float = 0.95, **kwargs
    ) -> pd.Series:
        """Calculate rolling VaR."""
        return returns.rolling(window).quantile(1 - confidence_level)

    @staticmethod
    def _expanding(
        returns: pd.Series,
        min_periods: int = 21,
        confidence_level: float = 0.95,
        **kwargs,
    ) -> pd.Series:
        """Calculate expanding VaR."""
        return returns.expanding(min_periods).quantile(1 - confidence_level)
