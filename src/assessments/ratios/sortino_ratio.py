from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.dataclasses.assessment import Assessment


@dataclass
class SortinoRatio(Assessment):
    def calc(self) -> float:
        self._downside: pd.Series = self.excess_returns[self.excess_returns < 0]
        self.downside_deviation = np.sqrt(np.mean(np.square(self._downside)))

        self.value: float = (
            float((self.excess_returns).mean() / self.downside_deviation * np.sqrt(252))
            if self.downside_deviation > 0
            else np.nan
        )
        return self.value
