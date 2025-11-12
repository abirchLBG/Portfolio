from dataclasses import dataclass

import pandas as pd

from src.assessments.base_assessment import BaseAssessment


@dataclass(kw_only=True)
class MaxDrawdown(BaseAssessment):
    def calc(self) -> float:
        cum_returns: pd.Series = self.config.returns.add(1).cumprod()
        running_max: pd.Series = cum_returns.cummax()

        drawdown: pd.Series = cum_returns.div(running_max).sub(1)
        self.value: float = float(drawdown.min())

        return self.value
