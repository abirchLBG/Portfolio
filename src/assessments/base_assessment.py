from abc import ABC
from dataclasses import dataclass

import pandas as pd

from src.dataclasses.assessment_config import AssessmentConfig
from src.dataclasses.results import AssessmentResults
from src.utils.timer import AssessmentTimer, timed_calc


@dataclass(kw_only=True)
class BaseAssessment(ABC):
    config: AssessmentConfig

    def __post_init__(self):
        self.results: AssessmentResults = AssessmentResults()
        self.timer = AssessmentTimer(self.__class__.__name__)

        self.summary = timed_calc(self.summary.__func__).__get__(self)
        self.rolling = timed_calc(self.rolling.__func__).__get__(self)
        self.expanding = timed_calc(self.expanding.__func__).__get__(self)

    def _repr_wrapper(self) -> str:
        return f"<{self.__class__.__name__}>"

    def __repr__(self) -> str:
        return self._repr_wrapper()

    @timed_calc
    def summary(self):
        pass

    @timed_calc
    def rolling(self):
        pass

    @timed_calc
    def expanding(self):
        pass

    def all(self) -> dict[str, float | pd.Series]:
        return {
            "summary": self.summary(),
            "rolling": self.rolling(),
            "expanding": self.expanding(),
        }
