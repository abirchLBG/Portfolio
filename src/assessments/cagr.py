from dataclasses import dataclass

import numpy as np

from src.dataclasses.assessment import Assessment


@dataclass
class CAGR(Assessment):
    def calc(self) -> float:
        self.value: float = float(
            np.prod(self.returns.add(1)) ** (252 / len(self.returns)) - 1
        )
        return self.value
