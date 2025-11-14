import pandas as pd
import yfinance as yf

from src.constants import YfTickers


DEFAULT_RFR: pd.Series = (
    yf.Ticker(YfTickers.US_3mo)
    .history("10y")["Close"]
    .tz_localize(None)
    .div(100)
    .add(1)
    .pow(1 / 91)
    .sub(1)
)
