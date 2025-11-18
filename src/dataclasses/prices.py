from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from tqdm import tqdm
import pandas as pd

from src.dataclasses.ticker import Ticker
from src.dataclasses.transactions import Transactions
from src.utils.executors import DummyExecutor


@dataclass
class Prices:
    transactions: Transactions
    pool: ThreadPoolExecutor | None = None

    def __post_init__(self):
        tickers = list(self.transactions.data.columns)
        self._ticker_dict: dict[str, Ticker] = {}
        prices_dict: dict[str, pd.Series] = {}

        def _load_ticker(ticker: str):
            """Build Ticker object and return its prices."""
            ticker_obj = Ticker(ticker)
            return ticker, ticker_obj, ticker_obj.prices

        use_pool = DummyExecutor() if self.pool is None else self.pool

        futures = [use_pool.submit(_load_ticker, t) for t in tickers]

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Loading prices"
        ):  # type: ignore
            ticker, ticker_obj, prices = future.result()
            self._ticker_dict[ticker] = ticker_obj
            prices_dict[ticker] = prices

        self.data = pd.concat(prices_dict, axis=1)
