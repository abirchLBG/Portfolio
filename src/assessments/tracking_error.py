from dataclasses import dataclass

import numpy as np

from src.dataclasses.assessment import Assessment


@dataclass
class TrackingError(Assessment):
    def calc(self) -> float:
        self.value = float(
                (self.returns - self.bmk_returns).std() * np.sqrt(252)
        )
        return self.value
