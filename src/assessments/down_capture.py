from dataclasses import dataclass
from typing import ClassVar

import pandas as pd

from src.assessments.base_assessment import BaseAssessment
from src.constants import AssessmentName


@dataclass(kw_only=True)
class DownCapture(BaseAssessment):
    """Down Capture Ratio Assessment

    Formula:
        Down_Capture = (Mean(R_p | R_bmk < 0) / Mean(R_bmk | R_bmk < 0)) * 100

    Description:
        Measures portfolio performance during periods when the benchmark is negative.
        Expressed as a percentage. 100% means the portfolio captures all downside.
        < 100% means the portfolio outperforms (loses less) in down markets.
        > 100% means the portfolio underperforms (loses more) in down markets.
    """

    name: ClassVar[AssessmentName] = AssessmentName.DownCapture

    @staticmethod
    def _summary(returns: pd.Series, bmk: pd.Series, **kwargs) -> float:
        """Calculate Down Capture Ratio."""
        down_market = bmk < 0
        if not down_market.any():
            return float("nan")

        portfolio_down = returns[down_market].mean()
        benchmark_down = bmk[down_market].mean()

        if benchmark_down == 0:
            return float("nan")

        return float((portfolio_down / benchmark_down) * 100)

    @staticmethod
    def _rolling(
        returns: pd.Series, bmk: pd.Series, window: int, **kwargs
    ) -> pd.Series:
        """Calculate rolling Down Capture Ratio using vectorized operations."""

        def down_capture_window(idx):
            """Calculate Down Capture for window ending at idx."""
            if idx < window - 1:
                return float("nan")

            start_idx = idx - window + 1
            port_vals = returns.iloc[start_idx : idx + 1]
            bmk_vals = bmk.iloc[start_idx : idx + 1]

            down_mask = bmk_vals < 0
            if not down_mask.any():
                return float("nan")

            port_mean = port_vals[down_mask].mean()
            bmk_mean = bmk_vals[down_mask].mean()

            if bmk_mean == 0 or pd.isna(bmk_mean):
                return float("nan")

            return (port_mean / bmk_mean) * 100

        # Apply calculation using index position
        result = pd.Series(
            [down_capture_window(i) for i in range(len(returns))], index=returns.index
        )

        return result

    @staticmethod
    def _expanding(
        returns: pd.Series, bmk: pd.Series, min_periods: int = 21, **kwargs
    ) -> pd.Series:
        """Calculate expanding Down Capture Ratio using vectorized operations."""

        def down_capture_window(idx):
            """Calculate Down Capture for expanding window ending at idx."""
            if idx < min_periods - 1:
                return float("nan")

            port_vals = returns.iloc[: idx + 1]
            bmk_vals = bmk.iloc[: idx + 1]

            down_mask = bmk_vals < 0
            if not down_mask.any():
                return float("nan")

            port_mean = port_vals[down_mask].mean()
            bmk_mean = bmk_vals[down_mask].mean()

            if bmk_mean == 0 or pd.isna(bmk_mean):
                return float("nan")

            return (port_mean / bmk_mean) * 100

        # Apply calculation using index position
        result = pd.Series(
            [down_capture_window(i) for i in range(len(returns))], index=returns.index
        )

        return result
