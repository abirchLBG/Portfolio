from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Self

from frozendict import frozendict

from src.assessments.beta import Beta
from src.assessments.cagr import CAGR
from src.assessments.max_drawdown import MaxDrawdown
from src.assessments.ratios.calmar_ratio import CalmarRatio
from src.assessments.ratios.information_ratio import InformationRatio
from src.assessments.ratios.sharpe_ratio import SharpeRatio
from src.assessments.tracking_error import TrackingError
from src.dataclasses.assessment_config import AssessmentConfig


class AssessmentName(StrEnum):
    Beta = "Beta"
    CAGR = "CAGR"
    MaxDrawdown = "Max Drawdown"
    TrackingError = "Tracking Error"

    SharpeRatio = "Sharpe Ratio"
    SortinoRatio = "Sortino Ratio"
    InformationRatio = "Information Ratio"
    CalmarRatio = "Calmar Ratio"
    # TreynorRatio = "Treynor Ratio"


ALL_ASSESSMENTS: frozendict[AssessmentName, Any] = frozendict(
    {
        AssessmentName.Beta: Beta,
        AssessmentName.CAGR: CAGR,
        AssessmentName.MaxDrawdown: MaxDrawdown,
        AssessmentName.TrackingError: TrackingError,
        AssessmentName.SharpeRatio: SharpeRatio,
        AssessmentName.InformationRatio: InformationRatio,
        AssessmentName.CalmarRatio: CalmarRatio,
    }
)


@dataclass
class FullEvaluation:
    config: AssessmentConfig
    assessments: frozendict[AssessmentName, Any] = ALL_ASSESSMENTS

    def __post_init__(self):
        self._configured_assessments: dict[AssessmentName, Any] = dict(
            map(lambda item: (item[0], item[1](self.config)), self.assessments.items())
        )

    def run(self) -> Self:
        self.results: frozendict[AssessmentName, float] = frozendict(
            map(
                lambda item: (item[0], item[1].calc()),
                self._configured_assessments.items(),
            )
        )
        return self
