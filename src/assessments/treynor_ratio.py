from dataclasses import dataclass

from src.assessments.base_assessment import BaseAssessment


@dataclass(kw_only=True)
class TreynorRatio(BaseAssessment):
    # def calc(self) -> float:
    #     beta: float = Beta(returns=self.returns, rfr=self.rfr, bmk_returns=self.bmk_returns).calc()

    #     self.value: float =

    #     return self.value
    pass
