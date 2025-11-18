from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from src.assessments.base_assessment import BaseAssessment
from src.constants import AssessmentName


@dataclass(kw_only=True)
class CAGR(BaseAssessment):
    name: ClassVar[AssessmentName] = AssessmentName.CAGR

    @staticmethod
    def _summary(returns: pd.Series, ann_factor: int = 252, **kwargs) -> float:
        return float(np.prod(returns + 1) ** (ann_factor / len(returns)) - 1)

    @staticmethod
    def _rolling(
        returns: pd.Series, window: int = 252, ann_factor: int = 252, **kwargs
    ) -> pd.Series:
        return returns.rolling(window=window).apply(
            CAGR._summary, args=(ann_factor,), raw=True
        )

    @staticmethod
    def _expanding(
        returns: pd.Series, min_periods: int = 21, ann_factor: int = 252, **kwargs
    ) -> pd.Series:
        return returns.expanding(min_periods).apply(
            CAGR._summary,
            args=(ann_factor,),
            raw=True,
        )
