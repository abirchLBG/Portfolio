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
from src.assessments.sortino_ratio import SortinoRatio
from src.assessments.tracking_error import TrackingError
from src.assessments.treynor_ratio import TreynorRatio
from src.assessments.volatility import Volatility
from src.assessments.benchmark_correlation import BenchmarkCorrelation
from src.assessments.var import VaR
from src.assessments.cvar import CVaR
from src.assessments.up_capture import UpCapture
from src.assessments.down_capture import DownCapture
from src.assessments.omega_ratio import OmegaRatio
from src.assessments.skewness import Skewness
from src.assessments.kurtosis import Kurtosis
from src.assessments.semi_variance import SemiVariance
from src.assessments.r_squared import RSquared
from src.assessments.m2_ratio import M2Ratio
from src.assessments.mean_return import MeanReturn
from src.assessments.cumulative_returns import CumulativeReturns
from src.assessments.ulcer_index import UlcerIndex
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
    AssessmentName.BenchmarkCorrelation: BenchmarkCorrelation,
    AssessmentName.VaR: VaR,
    AssessmentName.CVaR: CVaR,
    AssessmentName.UpCapture: UpCapture,
    AssessmentName.DownCapture: DownCapture,
    AssessmentName.SharpeRatio: SharpeRatio,
    AssessmentName.SortinoRatio: SortinoRatio,
    AssessmentName.InformationRatio: InformationRatio,
    AssessmentName.CalmarRatio: CalmarRatio,
    AssessmentName.TreynorRatio: TreynorRatio,
    AssessmentName.JensensAlpha: JensensAlpha,
    AssessmentName.OmegaRatio: OmegaRatio,
    AssessmentName.M2Ratio: M2Ratio,
    AssessmentName.Skewness: Skewness,
    AssessmentName.Kurtosis: Kurtosis,
    AssessmentName.SemiVariance: SemiVariance,
    AssessmentName.RSquared: RSquared,
    AssessmentName.UlcerIndex: UlcerIndex,
    AssessmentName.MeanReturn: MeanReturn,
    AssessmentName.CumulativeReturns: CumulativeReturns,
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
        num_assessments = len(self._assessments)
        num_assessment_types = len(self._assessment_types)
        num_configs = (
            len(self.config._returns_list)
            * len(self.config._rfr_list)
            * len(self.config._bmk_list)
        )
        executor_type = self._executor.__class__.__name__

        lines = [
            "Evaluation(",
            f"  assessments={num_assessments}",
            f"  assessment_types={num_assessment_types}",
            f"  configurations={num_configs}",
            f"  executor={executor_type}",
            f"  overlap_mode={self.config.overlap_mode.value}",
            ")",
        ]
        return "\n".join(lines)

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

        For configs with multiple returns/rfr/bmk, this will run all combinations
        and organize results in a multilevel structure.

        Returns:
            EvaluationResults: Object containing all assessment results and timing data
        """
        results: dict[
            str, dict[AssessmentName | str, dict[AssessmentType, float | pd.Series]]
        ] = {}
        timer: dict[str, dict[AssessmentName | str, dict[AssessmentType, float]]] = {}

        # Iterate over all config combinations
        for config_key, single_config in self.config.iter_configs():
            logger.info(f"Running assessments for configuration: {config_key}")

            # Initialize assessments for this specific config
            initialized_assessments: dict[AssessmentName, Any] = dict(
                map(
                    lambda item: (item[0], item[1](config=single_config)),
                    self._assessments.items(),
                )
            )

            config_results: dict[
                AssessmentName | str, dict[AssessmentType, float | pd.Series]
            ] = {}
            config_timer: dict[AssessmentName | str, dict[AssessmentType, float]] = {}

            futures = {}
            for name, assessment in initialized_assessments.items():
                for assessment_type in self._assessment_types:
                    if issubclass(type(self._executor), Executor):
                        future = self._executor.submit(assessment._run, assessment_type)
                        futures[future] = (name, assessment_type)
                    else:
                        output = assessment._run(assessment_type)
                        config_results.setdefault(name, {})[assessment_type] = output[
                            "result"
                        ]
                        config_timer.setdefault(name, {})[assessment_type] = output[
                            "time"
                        ]

            # Collect results from futures
            for future, (name, assessment_type) in futures.items():
                output = future.result()
                config_results.setdefault(name, {})[assessment_type] = output["result"]
                config_timer.setdefault(name, {})[assessment_type] = output["time"]

            results[config_key] = config_results
            timer[config_key] = config_timer

        return EvaluationResults(results=results, timer=timer, config=self.config)
