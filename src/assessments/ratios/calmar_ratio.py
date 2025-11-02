from dataclasses import dataclass


from src.assessments.max_drawdown import MaxDrawdown
from src.dataclasses.assessment_config import AssessmentConfig
from src.assessments.cagr import CAGR


@dataclass
class CalmarRatio:
    config: AssessmentConfig

    def calc(self) -> float:
        cagr: float = CAGR(config=self.config).calc()
        max_dd: float = MaxDrawdown(config=self.config).calc()

        self.value: float = cagr / abs(max_dd)
        return self.value
