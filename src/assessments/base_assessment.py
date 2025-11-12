from abc import ABC
from dataclasses import dataclass
from typing import Any, Callable

from src.dataclasses.assessment_config import AssessmentConfig


from time import perf_counter
from functools import wraps


def timed_calc(method: Callable) -> Any:
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        start = perf_counter()
        result = method(self, *args, **kwargs)
        self.calc_time = perf_counter() - start

        return result

    return wrapper


@dataclass(kw_only=True)
class BaseAssessment(ABC):
    config: AssessmentConfig

    def __post_init__(self):
        self.calc_time: float | None = None
        self.calc = timed_calc(self.calc.__func__).__get__(self)

    def _repr_override(self) -> str:
        return f"<{self.__class__.__name__}>"

    def __repr__(self) -> str:
        return self._repr_override()

    @timed_calc
    def calc(self):
        pass
