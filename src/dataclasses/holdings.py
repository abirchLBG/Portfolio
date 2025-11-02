import numpy as np
from src.dataclasses.prices import Prices
from src.dataclasses.subscriptions import Subscriptions
from src.dataclasses.transactions import Transactions


import pandas as pd
from dataclasses import dataclass


@dataclass
class Holdings:
    transactions: Transactions
    prices: Prices
    subscriptions: Subscriptions

    def __post_init__(self):
        self.data: pd.DataFrame = (
            (self.transactions.data.cumsum() * self.prices.data)
            .dropna()
            .resample("B")
            .first()
        )
        self.total = self.data.sum(axis=1).resample("B").first()

        # Money weighted returns
        self.mwr = self.total.pct_change().dropna().resample("B").first()
        self.money_weighted_returns = self.mwr

        # # Time weighted returns
        cash_flows: pd.Series = (
            self.subscriptions.data["USD"]
            .resample("D")
            .sum()  # in case of multiple deposits per day
            .reindex(self.total.index.union(self.subscriptions.data.index))
            .fillna(0.0)
        )

        V: pd.Series = self.total.reindex(cash_flows.index).ffill()
        V_prev: pd.Series = V.shift(1)

        # R_t = (V_t - cf_t - V_{t-1}) / V_{t-1}
        twr: pd.Series = (V - cash_flows - V_prev) / V_prev
        twr = twr.replace([np.inf, -np.inf], np.nan).dropna()

        self.twr: pd.Series = twr.resample("B").first()
        self.time_weighted_returns: pd.Series = self.twr
