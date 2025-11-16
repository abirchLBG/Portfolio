from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Self, Type

from frozendict import frozendict
from pyparsing import Iterable

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
from src.dataclasses.assessment_config import AssessmentConfig


from logging import Logger, getLogger

from src.dataclasses.assessment_results import AssessmentType

logger: Logger = getLogger(__name__)


class AssessmentName(StrEnum):
    Beta = "Beta"
    CAGR = "CAGR"
    MaxDrawdown = "Max Drawdown"
    TrackingError = "Tracking Error"

    SharpeRatio = "Sharpe Ratio"
    SortinoRatio = "Sortino Ratio"
    InformationRatio = "Information Ratio"
    CalmarRatio = "Calmar Ratio"
    TreynorRatio = "Treynor Ratio"
    JensensAlpha = "Jensen's Alpha"


ALL_ASSESSMENTS: frozendict[AssessmentName, Type[BaseAssessment]] = frozendict(
    {
        AssessmentName.Beta: Beta,
        AssessmentName.CAGR: CAGR,
        AssessmentName.MaxDrawdown: MaxDrawdown,
        AssessmentName.TrackingError: TrackingError,
        AssessmentName.SharpeRatio: SharpeRatio,
        AssessmentName.InformationRatio: InformationRatio,
        AssessmentName.CalmarRatio: CalmarRatio,
        AssessmentName.TreynorRatio: TreynorRatio,
        AssessmentName.JensensAlpha: JensensAlpha,
    }
)

ALL_ASSESSMENT_TYPES: frozenset[AssessmentType] = frozenset({v for v in AssessmentType})


@dataclass
class Evaluation:
    config: AssessmentConfig

    _assessments: frozendict[AssessmentName, Type[BaseAssessment]] = ALL_ASSESSMENTS
    _assessment_types: frozenset[AssessmentType] = ALL_ASSESSMENT_TYPES

    def __post_init__(self):
        self._results: dict[AssessmentName, float] | None = None
        self._timer: dict[AssessmentName, float] = {}

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
        self._assessments = frozendict(
            {name: ALL_ASSESSMENTS[name] for name in assessments}
        )

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

        logger.info("Running with filtered assessments")

        self._assessment_types = frozenset(assessment_types)

        return self

    def display_timer_stats(self) -> None:
        """Method to display the timer stats of the evaluation.

        Returns:
            str: Formatted string of timer stats.
        """
        if not len(self._timer):
            logger.warning("No timer stats to display. Run evaluation first.")
            return

        logger.info("Assessment Timing Breakdown:")
        logger.info("-" * 37)

        for name, assessment in self._initialized_assessments.items():
            time_taken: str = f"{assessment.calc_time:.4f}s"

            fmt_str = f"{name:{' '}<25}|    {time_taken}"  # 37 chars
            logger.info(fmt_str)

        logger.info("-" * 37)

    def run(self) -> Self:
        self._init_assessments()
        self._results = {}

        print(len(self._initialized_assessments))
        for name, assessment in self._initialized_assessments.items():
            for assessment_type in self._assessment_types:
                assessment.run(assessment_type)

            self._results[name] = assessment.results

        return self
