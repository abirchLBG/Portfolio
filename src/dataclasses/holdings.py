import numpy as np
from src.dataclasses.prices import Prices
from src.dataclasses.subscriptions import Subscriptions
from src.dataclasses.transactions import Transactions


import pandas as pd
from dataclasses import dataclass


def time_weighted_return(
    cumulative_holdings: pd.Series, cash_flows: pd.Series
) -> pd.Series:
    V: pd.Series = cumulative_holdings.reindex(cash_flows.index).ffill()
    V_prev: pd.Series = V.shift(1)

    # R_t = (V_t - cf_t - V_{t-1}) / V_{t-1}
    twr: pd.Series = (V - cash_flows - V_prev) / V_prev
    twr = twr.replace([np.inf, -np.inf], np.nan).dropna()
    twr = twr.resample("B").first()

    return twr


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
        self.total: pd.Series = self.data.sum(axis=1)
        self.weights: pd.DataFrame = self.data.div(self.total, axis=0)

        # Money weighted returns
        self.mwr: pd.Series = self.total.pct_change().dropna()
        self.money_weighted_returns: pd.Series = self.mwr

        # Time weighted returns
        self.twr: pd.Series = time_weighted_return(
            self.total, self.subscriptions.cash_flows
        )
        self.time_weighted_returns: pd.Series = self.twr
