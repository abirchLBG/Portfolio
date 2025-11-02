from abc import ABC
from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(kw_only=True)
class AssessmentConfig(ABC):
    returns: pd.Series
    rfr: pd.Series
    bmk: pd.Series

    start: str | pd.Timestamp | date | None = None
    end: str | pd.Timestamp | date | None = None

    def __post_init__(self):
        if self.start is not None:
            self.start = pd.Timestamp(self.start).date()
            self.returns = self.returns[self.start :]

        if self.end is not None:
            self.end = pd.Timestamp(self.end).date()
            self.returns = self.returns[: self.end]

        self.rfr = self.rfr.reindex(self.returns.index).fillna(0)
        self.bmk = self.bmk.reindex(self.returns.index).fillna(0)

        if self.rfr.isna().any():
            raise ValueError("Risk-free rate has missing data")

        if self.returns.isna().any():
            raise ValueError("Returns have missing data")

        if not self.returns.index.equals(self.rfr.index):
            raise ValueError("Index mismatch between returns and rfr")

        # Calculate excess
        self.excess_returns: pd.Series = self.returns - self.rfr
        self.active_returns: pd.Series = self.returns - self.bmk
