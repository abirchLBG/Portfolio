from dataclasses import dataclass
from typing import ClassVar

import pandas as pd

from src.assessments.base_assessment import BaseAssessment
from src.constants import AssessmentName


@dataclass(kw_only=True)
class CVaR(BaseAssessment):
    """Conditional Value at Risk (CVaR / Expected Shortfall) Assessment

    Formula:
        CVaR_α = E[R_p | R_p <= VaR_α]

    Description:
        Expected loss given that the loss exceeds VaR.
        Also known as Expected Shortfall (ES) or Average Value at Risk (AVaR).
        Measures tail risk beyond VaR threshold.
        Expressed as a negative number representing the expected loss.
        For example, 95% CVaR is the average of all losses worse than the 95% VaR.
    """

    name: ClassVar[AssessmentName] = AssessmentName.CVaR

    @staticmethod
    def _summary(returns: pd.Series, confidence_level: float = 0.95, **kwargs) -> float:
        """Calculate CVaR using historical simulation method."""
        var_threshold = returns.quantile(1 - confidence_level)
        # Average of all returns at or below the VaR threshold
        return float(returns[returns <= var_threshold].mean())

    @staticmethod
    def _rolling(
        returns: pd.Series, window: int, confidence_level: float = 0.95, **kwargs
    ) -> pd.Series:
        """Calculate rolling CVaR using vectorized operations."""

        def cvar_calc(x):
            """Optimized CVaR calculation for a window."""
            if len(x) < 2:
                return float("nan")
            threshold = x.quantile(1 - confidence_level)
            tail_losses = x[x <= threshold]
            return tail_losses.mean() if len(tail_losses) > 0 else float("nan")

        # Using pandas built-in rolling with apply is already optimized
        # The apply function runs on Series objects which use vectorized operations internally
        return returns.rolling(window).apply(cvar_calc, raw=False)

    @staticmethod
    def _expanding(
        returns: pd.Series,
        min_periods: int = 21,
        confidence_level: float = 0.95,
        **kwargs,
    ) -> pd.Series:
        """Calculate expanding CVaR using vectorized operations."""

        def cvar_calc(x):
            """Optimized CVaR calculation for expanding window."""
            if len(x) < 2:
                return float("nan")
            threshold = x.quantile(1 - confidence_level)
            tail_losses = x[x <= threshold]
            return tail_losses.mean() if len(tail_losses) > 0 else float("nan")

        # Using pandas built-in expanding with apply is already optimized
        return returns.expanding(min_periods).apply(cvar_calc, raw=False)
