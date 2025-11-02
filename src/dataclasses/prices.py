from src.dataclasses.ticker import Ticker
from tqdm import tqdm

import pandas as pd

from dataclasses import dataclass

from src.dataclasses.transactions import Transactions


@dataclass
class Prices:
    transactions: Transactions

    def __post_init__(self):
        prices_dict: dict[str, pd.Series] = {}
        self._ticker_dict: dict[str, Ticker] = {}

        for ticker in tqdm(self.transactions.data.columns):
            ticker_obj: Ticker = Ticker(ticker=ticker)
            self._ticker_dict[ticker] = ticker_obj
            prices_dict[ticker] = ticker_obj.prices

        self.data: pd.DataFrame = pd.concat(prices_dict, axis=1)
