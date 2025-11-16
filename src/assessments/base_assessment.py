from abc import ABC
from dataclasses import dataclass

import pandas as pd

from src.dataclasses.assessment_config import AssessmentConfig
from src.dataclasses.assessment_results import AssessmentResults, AssessmentType
from src.utils.timer import AssessmentTimer, timed_calc


@dataclass(kw_only=True)
class BaseAssessment(ABC):
    config: AssessmentConfig

    def __post_init__(self):
        self.results: AssessmentResults = AssessmentResults()
        self.timer = AssessmentTimer(self.__class__.__name__)

        self._child_cls = type(self)

    def _repr_wrapper(self) -> str:
        return f"{self.__class__.__name__}()"

    def __repr__(self) -> str:
        return self._repr_wrapper()

    @staticmethod
    def _summary():
        raise NotImplementedError()

    @staticmethod
    def _rolling():
        raise NotImplementedError()

    @staticmethod
    def _expanding():
        raise NotImplementedError()

    @timed_calc
    def summary(self):
        return self._child_cls._summary(**self.config.kwargs)

    @timed_calc
    def rolling(self):
        return self._child_cls._rolling(**self.config.kwargs)

    @timed_calc
    def expanding(self):
        return self._child_cls._expanding(**self.config.kwargs)

    def all(self) -> dict[str | AssessmentType, float | pd.Series]:
        return {name: self.run(name) for name in AssessmentType}

    def run(self, name: AssessmentType | str) -> float | pd.Series:
        return getattr(self, name)()
