from dataclasses import dataclass

import numpy as np

from src.dataclasses.assessment import Assessment


@dataclass
class Beta(Assessment):
    def calc(self) -> float:
        cov: float = np.cov(self.returns, self.bmk_returns)[0, 1]
        self.value: float = float(cov / np.var(self.bmk_returns))

        return self.value
