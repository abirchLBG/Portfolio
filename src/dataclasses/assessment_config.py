from abc import ABC
from dataclasses import asdict, dataclass
from datetime import date
import logging

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class AssessmentConfig(ABC):
    returns: pd.Series
    rfr: pd.Series
    bmk: pd.Series

    start: str | pd.Timestamp | date | None = None
    end: str | pd.Timestamp | date | None = None

    ann_factor: int = 252
    window: int = 252
    min_periods: int = 21  # 1 BMonth

    def __post_init__(self):
        # Validate numeric parameters BEFORE any data processing
        if self.ann_factor <= 0:
            raise ValueError(f"ann_factor must be positive, got {self.ann_factor}")

        if self.window <= 0:
            raise ValueError(f"window must be positive, got {self.window}")

        if self.min_periods <= 0:
            raise ValueError(f"min_periods must be positive, got {self.min_periods}")

        if self.min_periods > self.window:
            logger.warning(
                f"min_periods ({self.min_periods}) is greater than window ({self.window}). "
                "This may cause issues in rolling calculations."
            )

        # Validate input series are not empty
        if self.returns.empty:
            raise ValueError("returns series cannot be empty")

        if self.rfr.empty:
            raise ValueError("rfr series cannot be empty")

        if self.bmk.empty:
            raise ValueError("bmk series cannot be empty")

        # Validate minimum data length
        min_required_length = max(
            self.min_periods, 2
        )  # Need at least 2 points for calculations
        if len(self.returns) < min_required_length:
            raise ValueError(
                f"returns must have at least {min_required_length} data points, "
                f"got {len(self.returns)}"
            )

        # Now proceed with data filtering
        if self.start is not None:
            self.start = pd.Timestamp(self.start).date()
            self.returns = self.returns[self.start :]

        if self.end is not None:
            self.end = pd.Timestamp(self.end).date()
            self.returns = self.returns[: self.end]

        self.rfr = self.rfr.reindex(self.returns.index).fillna(0)
        self.bmk = self.bmk.reindex(self.returns.index).fillna(0)

        self.kwargs: dict = asdict(self)

        if self.rfr.isna().any():
            raise ValueError("Risk-free rate has missing data")

        if self.returns.isna().any():
            raise ValueError("Returns have missing data")

        if not self.returns.index.equals(self.rfr.index):
            raise ValueError("Index mismatch between returns and rfr")
