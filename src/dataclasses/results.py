from dataclasses import dataclass

import pandas as pd


@dataclass(kw_only=True)
class AssessmentResults:
    summary: float | None = None
    rolling: float | pd.Series | None = None
    expanding: float | pd.Series | None = None

    def update(self, name: str, value: float | pd.Series) -> None:
        setattr(self, name, value)
