from dataclasses import dataclass

import numpy as np

from src.assessments.base_assessment import BaseAssessment


@dataclass(kw_only=True)
class TrackingError(BaseAssessment):
    def calc(self) -> float:
        self.value = float(self.config.active_returns.std() * np.sqrt(252))

        return self.value
