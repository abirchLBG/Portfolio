from dataclasses import dataclass
from typing import ClassVar

import pandas as pd

from src.assessments.base_assessment import BaseAssessment
from src.constants import AssessmentName


@dataclass(kw_only=True)
class UpCapture(BaseAssessment):
    """Up Capture Ratio Assessment

    Formula:
        Up_Capture = (Mean(R_p | R_bmk > 0) / Mean(R_bmk | R_bmk > 0)) * 100

    Description:
        Measures portfolio performance during periods when the benchmark is positive.
        Expressed as a percentage. 100% means the portfolio captures all upside.
        > 100% means the portfolio outperforms in up markets.
        < 100% means the portfolio underperforms in up markets.
    """

    name: ClassVar[AssessmentName] = AssessmentName.UpCapture

    @staticmethod
    def _summary(returns: pd.Series, bmk: pd.Series, **kwargs) -> float:
        """Calculate Up Capture Ratio."""
        up_market = bmk > 0
        if not up_market.any():
            return float("nan")

        portfolio_up = returns[up_market].mean()
        benchmark_up = bmk[up_market].mean()

        if benchmark_up == 0:
            return float("nan")

        return float((portfolio_up / benchmark_up) * 100)

    @staticmethod
    def _rolling(
        returns: pd.Series, bmk: pd.Series, window: int, **kwargs
    ) -> pd.Series:
        """Calculate rolling Up Capture Ratio using vectorized operations."""

        def up_capture_window(idx):
            """Calculate Up Capture for window ending at idx."""
            if idx < window - 1:
                return float("nan")

            start_idx = idx - window + 1
            port_vals = returns.iloc[start_idx : idx + 1]
            bmk_vals = bmk.iloc[start_idx : idx + 1]

            up_mask = bmk_vals > 0
            if not up_mask.any():
                return float("nan")

            port_mean = port_vals[up_mask].mean()
            bmk_mean = bmk_vals[up_mask].mean()

            if bmk_mean == 0 or pd.isna(bmk_mean):
                return float("nan")

            return (port_mean / bmk_mean) * 100

        # Apply calculation using index position
        result = pd.Series(
            [up_capture_window(i) for i in range(len(returns))], index=returns.index
        )

        return result

    @staticmethod
    def _expanding(
        returns: pd.Series, bmk: pd.Series, min_periods: int = 21, **kwargs
    ) -> pd.Series:
        """Calculate expanding Up Capture Ratio using vectorized operations."""

        def up_capture_window(idx):
            """Calculate Up Capture for expanding window ending at idx."""
            if idx < min_periods - 1:
                return float("nan")

            port_vals = returns.iloc[: idx + 1]
            bmk_vals = bmk.iloc[: idx + 1]

            up_mask = bmk_vals > 0
            if not up_mask.any():
                return float("nan")

            port_mean = port_vals[up_mask].mean()
            bmk_mean = bmk_vals[up_mask].mean()

            if bmk_mean == 0 or pd.isna(bmk_mean):
                return float("nan")

            return (port_mean / bmk_mean) * 100

        # Apply calculation using index position
        result = pd.Series(
            [up_capture_window(i) for i in range(len(returns))], index=returns.index
        )

        return result
