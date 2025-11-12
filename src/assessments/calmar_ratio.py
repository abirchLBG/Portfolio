from dataclasses import dataclass


from src.assessments.base_assessment import BaseAssessment
from src.assessments.max_drawdown import MaxDrawdown
from src.assessments.cagr import CAGR


@dataclass(kw_only=True)
class CalmarRatio(BaseAssessment):
    def calc(self) -> float:
        cagr: float = CAGR(config=self.config).calc()
        max_dd: float = MaxDrawdown(config=self.config).calc()

        self.value: float = cagr / abs(max_dd)
        return self.value
