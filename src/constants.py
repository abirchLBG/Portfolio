from enum import StrEnum


class YfTickers(StrEnum):
    # ccy
    GBPUSD = "GBPUSD=X"

    # rfr
    US_3mo = "^IRX"
    US_10yr = "^TNX"

    # bmk
    SPX = "^SPX"
    QQQ = "QQQ"


class AssessmentName(StrEnum):
    Beta = "Beta"
    CAGR = "CAGR"
    MaxDrawdown = "Max Drawdown"
    TrackingError = "Tracking Error"
    Volatility = "Volatility"
    BenchmarkCorrelation = "Benchmark Correlation"
    VaR = "VaR"
    CVaR = "CVaR"
    UpCapture = "Up Capture"
    DownCapture = "Down Capture"

    SharpeRatio = "Sharpe Ratio"
    SortinoRatio = "Sortino Ratio"
    InformationRatio = "Information Ratio"
    CalmarRatio = "Calmar Ratio"
    TreynorRatio = "Treynor Ratio"
    JensensAlpha = "Jensen's Alpha"
    OmegaRatio = "Omega Ratio"
    M2Ratio = "M2 Ratio"

    # Statistical Measures
    Skewness = "Skewness"
    Kurtosis = "Kurtosis"
    SemiVariance = "Semi-Variance"
    RSquared = "R-Squared"
    UlcerIndex = "Ulcer Index"

    # Return Metrics
    MeanReturn = "Mean Return"
    AnnualizedReturns = "Annualized Returns"
    CumulativeReturns = "Cumulative Returns"
