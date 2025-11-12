import threading
import pandas as pd
from src.constants import YfTickers
import yfinance as yf
from concurrent.futures import Executor, Future, _base

DEFAULT_RFR: pd.Series = (
    yf.Ticker(YfTickers.US_3mo)
    .history("10y")["Close"]
    .tz_localize(None)
    .div(100)
    .add(1)
    .pow(1 / 91)
    .sub(1)
)


class DummyExecutor(Executor):
    def submit(self, fn, *args, **kwargs):
        class DummyFuture(Future):
            def __init__(self):
                self._condition = threading.Condition()
                self._state = _base.FINISHED
                self._waiters = []

            def result(self):
                return fn(*args, **kwargs)

        return DummyFuture()
