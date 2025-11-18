from abc import ABC
from dataclasses import dataclass
from time import perf_counter
from typing import ClassVar
import pandas as pd

from src.dataclasses.assessment_config import AssessmentConfig
from src.dataclasses.assessment_results import AssessmentType


@dataclass(kw_only=True)
class BaseAssessment(ABC):
    config: AssessmentConfig
    name: ClassVar[str]

    def __post_init__(self):
        self._child_cls = type(self)

    @staticmethod
    def _summary() -> float:
        raise NotImplementedError()

    @staticmethod
    def _rolling() -> pd.Series:
        raise NotImplementedError()

    @staticmethod
    def _expanding() -> pd.Series:
        raise NotImplementedError()

    def summary(self) -> float:
        return self._child_cls._summary(**self.config.kwargs)

    def rolling(self) -> pd.Series:
        return self._child_cls._rolling(**self.config.kwargs)

    def expanding(self) -> pd.Series:
        return self._child_cls._expanding(**self.config.kwargs)

    def _run(
        self, assessment_type: AssessmentType | str = AssessmentType.Summary
    ) -> dict:
        start: float = perf_counter()
        result = getattr(self, assessment_type)()
        elapsed: float = perf_counter() - start

        return {
            "assessment": self.name,
            "type": assessment_type,
            "result": result,
            "time": elapsed,
        }
