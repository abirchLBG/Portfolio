from dataclasses import dataclass

import numpy as np

from src.assessments.base_assessment import BaseAssessment


@dataclass(kw_only=True)
class Beta(BaseAssessment):
    def calc(self) -> float:
        cov: float = np.cov(self.config.returns, self.config.bmk)[0, 1]
        self.value: float = float(cov / np.var(self.config.bmk))

        return self.value
