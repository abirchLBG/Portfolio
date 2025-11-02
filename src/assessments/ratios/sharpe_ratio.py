from dataclasses import dataclass

import numpy as np

from src.dataclasses.assessment import Assessment


@dataclass
class SharpeRatio(Assessment):
    def calc(self) -> float:
        excess_std: float = self.excess_returns.std()
        self.value: float = (
            float((self.excess_returns).mean() / excess_std * np.sqrt(252))
            if excess_std > 0
            else np.nan
        )
        return self.value
