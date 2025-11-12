from dataclasses import dataclass

import numpy as np

from src.assessments.base_assessment import BaseAssessment
from src.assessments.tracking_error import TrackingError


@dataclass(kw_only=True)
class InformationRatio(BaseAssessment):
    def calc(self) -> float:
        tracking_error: float = TrackingError(config=self.config).calc()
        mean_active: float = self.config.active_returns.mean() * 252

        self.value: float = (
            float(mean_active / tracking_error) if tracking_error != 0 else np.nan
        )

        return self.value
