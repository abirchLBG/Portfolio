from dataclasses import dataclass

import numpy as np

from src.dataclasses.assessment_config import AssessmentConfig


@dataclass
class Beta:
    config: AssessmentConfig

    def calc(self) -> float:
        cov: float = np.cov(self.config.returns, self.config.bmk)[0, 1]
        self.value: float = float(cov / np.var(self.config.bmk))

        return self.value
