from enum import StrEnum


class YfTickers(StrEnum):
    US_3mo = "^IRX"
    GBPUSD = "GBPUSD=X"


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
