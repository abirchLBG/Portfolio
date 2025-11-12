from dataclasses import dataclass

import pandas as pd
import yfinance as yf


@dataclass
class Ticker:
    ticker: str
    is_fx: bool = False

    def __post_init__(self) -> None:
        self.yf_ticker = yf.Ticker(self.ticker)

        self.info: dict[str, str | int] = self._get_fast_info()
        self.ccy: str = str(self.info["currency"])

        self.prices: pd.Series = self._get_price_history()
        self.returns = (
            self.prices.pct_change(fill_method=None).dropna(how="any").rename("Returns")
        )

    def _get_fast_info(self) -> dict[str, str | int]:
        if hasattr(self, "info"):
            return self.info
        return dict(self.yf_ticker.fast_info)

    def _get_price_history(self, *, n_years: int = 10) -> pd.Series:
        if hasattr(self, "prices"):
            return self.prices

        close: pd.Series = self.yf_ticker.history(period=f"{n_years}y")["Close"]
        close.index = close.index.tz_localize(None)  # type: ignore
        close = close.resample("D").ffill().rename(self.ticker)
        self._raw_close: pd.Series = close.copy()

        if self.is_fx:
            return close

        if self.ccy != "USD":
            close = close * Ticker(f"{self.ccy}USD=X", is_fx=True).prices

            if self.ccy == "GBp":
                close = close / 100

        return close
