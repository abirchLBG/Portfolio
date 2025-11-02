import pandas as pd
from src.constants import YfTickers
import yfinance as yf

DEFAULT_RFR: pd.Series = (
    yf.Ticker(YfTickers.US_3mo)
    .history("10y")["Close"]
    .tz_localize(None)
    .div(100)
    .add(1)
    .pow(1 / 91)
    .sub(1)
)
