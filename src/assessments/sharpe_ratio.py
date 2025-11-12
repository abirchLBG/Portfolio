from dataclasses import dataclass

import numpy as np

from src.assessments.base_assessment import BaseAssessment


@dataclass(kw_only=True)
class SharpeRatio(BaseAssessment):
    def calc(self) -> float:
        excess_std: float = self.config.excess_returns.std()

        self.value: float = (
            float(((self.config.excess_returns).mean() * np.sqrt(252)) / (excess_std))
            if excess_std > 0
            else np.nan
        )

        return self.value
