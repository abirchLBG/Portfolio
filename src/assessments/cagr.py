from dataclasses import dataclass

import numpy as np

from src.dataclasses.assessment_config import AssessmentConfig


@dataclass
class CAGR:
    config: AssessmentConfig

    def calc(self) -> float:
        self.value: float = float(
            np.prod(self.config.returns.add(1)) ** (252 / len(self.config.returns)) - 1
        )
        return self.value
