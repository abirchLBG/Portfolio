from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

# from src.constants import AssessmentName


class AssessmentType(StrEnum):
    Summary = "summary"
    Rolling = "rolling"
    Expanding = "expanding"


@dataclass
class AssessmentResults:
    summary: float | None = None
    rolling: float | pd.Series | None = None
    expanding: float | pd.Series | None = None

    def update(self, name: str, value: float | pd.Series) -> None:
        setattr(self, name, value)


# @dataclass
# class CombinedSummaryResults:
#     results: dict[AssessmentName, float]

# @dataclass
# class CombinedRollingResults:
#     results: dict[AssessmentName, pd.Series]

# @dataclass
# class CombinedExpandingResults:
#     results: dict[AssessmentName, pd.Series]

# @dataclass
# class EvaluationResults:
#     assessment_results: Iterable[AssessmentResults]
