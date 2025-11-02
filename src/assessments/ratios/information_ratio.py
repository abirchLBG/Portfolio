from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.assessments.tracking_error import TrackingError
from src.dataclasses.assessment import Assessment


@dataclass
class InformationRatio(Assessment):
    def calc(self) -> float:
        tracking_error: float = TrackingError(returns=self.returns, rfr=self.rfr, bmk_returns=self.bmk_returns).calc()
        
        active_return: pd.Series = (self.returns - self.bmk_returns)
        mean_active: float = active_return.mean() * 252
        
        self.value: float = float(
            mean_active / tracking_error
        ) if tracking_error != 0 else np.nan

        return self.value
