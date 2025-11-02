from dataclasses import dataclass

import numpy as np

from src.dataclasses.assessment_config import AssessmentConfig


@dataclass
class TrackingError:
    config: AssessmentConfig

    def calc(self) -> float:
        self.value = float(self.config.active_returns.std() * np.sqrt(252))

        return self.value
