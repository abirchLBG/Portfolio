from dataclasses import dataclass
import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class TickerDataError(Exception):
    """Raised when ticker data cannot be fetched or is invalid."""

    pass


@dataclass
class Ticker:
    ticker: str
    is_fx: bool = False
    name: str | None = None  # Optional custom name for returns series
    _max_fx_depth: int = 5  # Prevent infinite recursion in FX conversion

    def __post_init__(self) -> None:
        # Use custom name if provided, otherwise use ticker symbol
        if self.name is None:
            self.name = self.ticker
        try:
            self.yf_ticker = yf.Ticker(self.ticker)
        except Exception as e:
            raise TickerDataError(
                f"Failed to create yfinance Ticker for {self.ticker}: {e}"
            )

        self.info: dict[str, str | int] = self._get_fast_info()
        self.ccy: str = str(self.info.get("currency", "USD"))

        self.prices: pd.Series = self._get_price_history()

        if self.prices.empty:
            raise TickerDataError(f"No price data available for {self.ticker}")

        self.returns = (
            self.prices.pct_change(fill_method=None).dropna(how="any").rename(self.name)
        )

    def _get_fast_info(self) -> dict[str, str | int]:
        if hasattr(self, "info"):
            return self.info

        try:
            info = dict(self.yf_ticker.fast_info)
            if not info:
                logger.warning(f"Empty fast_info for {self.ticker}, using defaults")
                return {"currency": "USD"}
            return info
        except Exception as e:
            logger.error(f"Failed to fetch fast_info for {self.ticker}: {e}")
            raise TickerDataError(f"Failed to fetch ticker info for {self.ticker}: {e}")

    def _get_price_history(self, *, n_years: int = 10, _fx_depth: int = 0) -> pd.Series:
        if hasattr(self, "prices"):
            return self.prices

        # Prevent infinite recursion in FX conversion
        if _fx_depth > self._max_fx_depth:
            raise TickerDataError(
                f"Maximum FX conversion depth ({self._max_fx_depth}) exceeded for {self.ticker}. "
                "Possible circular currency reference."
            )

        try:
            history = self.yf_ticker.history(period=f"{n_years}y")
            if history.empty:
                raise TickerDataError(f"No historical data returned for {self.ticker}")

            if "Close" not in history.columns:
                raise TickerDataError(f"No 'Close' price data for {self.ticker}")

            close: pd.Series = history["Close"]
        except Exception as e:
            logger.error(f"Failed to fetch price history for {self.ticker}: {e}")
            raise TickerDataError(
                f"Failed to fetch price history for {self.ticker}: {e}"
            )

        try:
            close.index = close.index.tz_localize(None)  # type: ignore
            close = close.resample("D").ffill().rename(self.ticker)
            self._raw_close: pd.Series = close.copy()
        except Exception as e:
            logger.error(f"Failed to process price data for {self.ticker}: {e}")
            raise TickerDataError(
                f"Failed to process price data for {self.ticker}: {e}"
            )

        if self.is_fx:
            return close

        # Convert to USD if needed
        if self.ccy != "USD":
            try:
                fx_ticker_symbol = f"{self.ccy}USD=X"
                logger.debug(
                    f"Converting {self.ticker} from {self.ccy} to USD using {fx_ticker_symbol}"
                )

                fx_ticker = Ticker(fx_ticker_symbol, is_fx=True)
                # Pass depth to prevent infinite recursion
                fx_ticker._get_price_history(_fx_depth=_fx_depth + 1)

                close = close * fx_ticker.prices

                # Handle GBp (pence) to GBP conversion
                if self.ccy == "GBp":
                    close = close / 100

            except Exception as e:
                logger.error(
                    f"Failed to convert {self.ticker} from {self.ccy} to USD: {e}"
                )
                raise TickerDataError(
                    f"Failed to convert {self.ticker} from {self.ccy} to USD: {e}"
                )

        return close
