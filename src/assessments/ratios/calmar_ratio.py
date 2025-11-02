from dataclasses import dataclass


from src.assessments.max_drawdown import MaxDrawdown
from src.dataclasses.assessment import Assessment
from src.assessments.cagr import CAGR


@dataclass
class CalmarRatio(Assessment):
    def calc(self) -> float:
        cagr: float = CAGR(
            returns=self.returns, rfr=self.rfr, bmk_returns=self.bmk_returns
        ).calc()
        max_dd: float = MaxDrawdown(
            returns=self.returns, rfr=self.rfr, bmk_returns=self.bmk_returns
        ).calc()

        self.value: float = cagr / abs(max_dd)
        return self.value
