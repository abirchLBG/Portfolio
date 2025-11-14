from dataclasses import dataclass
from functools import wraps
from time import perf_counter
from typing import Any, Callable
import pandas as pd


def timed_calc(method: Callable) -> Any:
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        start: float = perf_counter()
        result: Any = method(self, *args, **kwargs)
        elapsed: float = perf_counter() - start

        self.timer.update(method, elapsed)
        self.results.update(method.__name__, result)

        return result

    return wrapper


@dataclass
class AssessmentTimer:
    assessment_name: str

    def __post_init__(self):
        self._timings: dict[str, float] = {}
        self._call_count: dict[str, int] = {}

    def update(self, method: Callable, elapsed: float) -> None:
        name: str = method.__name__
        self._timings[name] = self._timings.get(name, 0) + elapsed
        self._call_count[name] = self._call_count.get(name, 0) + 1

    @property
    def _timing_dict(self) -> dict[str, float]:
        sum_all: float = sum(self._timings.values())
        return {**self._timings, "total": sum_all}

    @property
    def _call_count_dict(self) -> dict[str, int]:
        sum_all: int = sum(self._call_count.values())
        return {**self._call_count, "total": sum_all}

    @property
    def timing_table(self) -> pd.Series:
        return pd.Series(self._timing_dict, name="Time (s)")

    @property
    def call_count_table(self) -> pd.Series:
        return pd.Series(self._call_count_dict, name="Count")

    @property
    def df(self) -> pd.DataFrame:
        out = pd.DataFrame(
            {
                self.timing_table.name: self.timing_table,
                self.call_count_table.name: self.call_count_table,
            },
        )
        out.index = pd.MultiIndex.from_product(
            [[self.assessment_name], self.timing_table.index],
        )

        out["Mean (s)"] = out["Time (s)"] / out["Count"]
        out = out[[self.timing_table.name, "Mean (s)", self.call_count_table.name]]

        return out
