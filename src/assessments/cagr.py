from dataclasses import dataclass

import numpy as np

from src.assessments.base_assessment import BaseAssessment


@dataclass(kw_only=True)
class CAGR(BaseAssessment):
    def calc(self) -> float:
        self.value: float = float(
            np.prod(self.config.returns.add(1)) ** (252 / len(self.config.returns)) - 1
        )
        return self.value
