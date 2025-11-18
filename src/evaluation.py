from dataclasses import dataclass
from enum import Enum
from typing import Any, Self, Type
from concurrent.futures import Executor, ProcessPoolExecutor

import pandas as pd
from typing import Iterable

from src.assessments.base_assessment import BaseAssessment
from src.assessments.beta import Beta
from src.assessments.cagr import CAGR
from src.assessments.jensens_alpha import JensensAlpha
from src.assessments.max_drawdown import MaxDrawdown
from src.assessments.calmar_ratio import CalmarRatio
from src.assessments.information_ratio import InformationRatio
from src.assessments.sharpe_ratio import SharpeRatio
from src.assessments.tracking_error import TrackingError
from src.assessments.treynor_ratio import TreynorRatio
from src.assessments.volatility import Volatility
from src.assessments.correlation import Correlation
from src.assessments.var import ValueAtRisk
from src.assessments.cvar import CVaR
from src.assessments.up_capture import UpCapture
from src.assessments.down_capture import DownCapture
from src.constants import AssessmentName
from src.dataclasses.assessment_config import AssessmentConfig


from logging import Logger, getLogger

from src.dataclasses.assessment_results import AssessmentType, EvaluationResults
from src.utils.executors import DummyExecutor, RQExecutor

logger: Logger = getLogger(__name__)


ALL_ASSESSMENTS: dict[AssessmentName, Type[BaseAssessment]] = {
    AssessmentName.Beta: Beta,
    AssessmentName.CAGR: CAGR,
    AssessmentName.MaxDrawdown: MaxDrawdown,
    AssessmentName.TrackingError: TrackingError,
    AssessmentName.Volatility: Volatility,
    AssessmentName.Correlation: Correlation,
    AssessmentName.ValueAtRisk: ValueAtRisk,
    AssessmentName.CVaR: CVaR,
    AssessmentName.UpCapture: UpCapture,
    AssessmentName.DownCapture: DownCapture,
    AssessmentName.SharpeRatio: SharpeRatio,
    AssessmentName.InformationRatio: InformationRatio,
    AssessmentName.CalmarRatio: CalmarRatio,
    AssessmentName.TreynorRatio: TreynorRatio,
    AssessmentName.JensensAlpha: JensensAlpha,
}


ALL_ASSESSMENT_TYPES: frozenset[AssessmentType] = frozenset({v for v in AssessmentType})


class ExecutorType(Enum):
    DEFAULT = DummyExecutor
    ProcessPool = ProcessPoolExecutor
    Remote = RQExecutor

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)


@dataclass
class Evaluation:
    config: AssessmentConfig

    def __post_init__(self):
        self._assessments: dict[AssessmentName, Type[BaseAssessment]] = ALL_ASSESSMENTS
        self._assessment_types: list[AssessmentType] = list(AssessmentType)
        self._executor: DummyExecutor | ProcessPoolExecutor | RQExecutor = (
            ExecutorType.DEFAULT()
        )

    def __repr__(self) -> str:
        return "Evaluation"

    def _init_assessments(self) -> None:
        """Wrapper func to init the assessments."""
        logger.debug("Initializing assessments.")
        self._initialized_assessments: dict[AssessmentName, Any] = dict(
            map(
                lambda item: (item[0], item[1](config=self.config)),
                self._assessments.items(),
            )
        )

    def with_assessments(
        self, assessments: Iterable[AssessmentName] | None = None
    ) -> Self:
        """Method to change the Evaluation object to use filtered assessments from AssessmentName enum.

        Args:
            assessments (set[AssessmentName] | None, optional): Assessments to run the evaluation with. Defaults to None.

        Returns:
            Evaluation: Evaluation object with filtered assessments.
        """
        if not assessments:
            return self

        logger.info("Running with filtered assessments")
        self._assessments = {name: ALL_ASSESSMENTS[name] for name in assessments}

        return self

    def with_assessment_types(
        self, assessment_types: Iterable[AssessmentType] | None = None
    ) -> Self:
        """Method to change the Evaluation object to use filtered assessment types from AssessmentType enum.

        Args:
            assessments (set[AssessmentName] | None, optional): Assessments to run the evaluation with. Defaults to None.

        Returns:
            Evaluation: Evaluation object with filtered assessments.
        """
        if not assessment_types:
            return self

        logger.info("Running with filtered assessment types")

        self._assessment_types = list(assessment_types)

        return self

    def with_executor(
        self, executor: DummyExecutor | ProcessPoolExecutor | RQExecutor
    ) -> Self:
        logger.info(f"Using {executor.__class__.__name__}")
        self._executor = executor

        return self

    def run(self) -> EvaluationResults:
        """
        Run all configured assessments and return results.

        Returns:
            EvaluationResults: Object containing all assessment results and timing data
        """
        self._init_assessments()
        results: dict[
            AssessmentName | str, dict[AssessmentType, float | pd.Series]
        ] = {}
        timer: dict[AssessmentName | str, dict[AssessmentType, float]] = {}

        futures = {}
        for name, assessment in self._initialized_assessments.items():
            for assessment_type in self._assessment_types:
                if issubclass(type(self._executor), Executor):
                    future = self._executor.submit(assessment._run, assessment_type)
                    futures[future] = (name, assessment_type)
                else:
                    output = assessment._run(assessment_type)
                    results.setdefault(name, {})[assessment_type] = output["result"]
                    timer.setdefault(name, {})[assessment_type] = output["time"]

        # Collect results from futures
        for future, (name, assessment_type) in futures.items():
            output = future.result()
            results.setdefault(name, {})[assessment_type] = output["result"]
            timer.setdefault(name, {})[assessment_type] = output["time"]

        return EvaluationResults(results=results, timer=timer)
