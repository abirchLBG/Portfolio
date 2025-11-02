from dataclasses import dataclass

import pandas as pd

from src.dataclasses.assessment import Assessment


@dataclass
class MaxDrawdown(Assessment):
    def calc(self) -> float:
        cum_returns: pd.Series = self.returns.add(1).cumprod()
        running_max: pd.Series = cum_returns.cummax()

        drawdown: pd.Series = cum_returns.div(running_max).sub(1)
        self.value: float = float(drawdown.min())
        return self.value
