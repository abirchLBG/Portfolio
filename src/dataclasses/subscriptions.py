from dataclasses import dataclass
from enum import StrEnum
import pandas as pd

from src.constants import YfTickers
from src.dataclasses.ticker import Ticker


class SubCol(StrEnum):
    # Raw
    Portfolio = "Portfolio"
    Date = "Date"
    GBP = "GBP"

    # Derived
    USD = "USD"

    @classmethod
    def raw_cols(cls) -> list["SubCol"]:
        return [SubCol.Portfolio, SubCol.Date, SubCol.GBP]


@dataclass
class Subscriptions:
    data: pd.DataFrame

    def __post_init__(self):
        # Time weighted returns
        self.cash_flows: pd.Series = (
            self.data["USD"]
            .resample("D")
            .sum()  # in case of multiple deposits per day
            .fillna(0.0)
        )

    @classmethod
    def from_csv(cls, path: str) -> "Subscriptions":
        df: pd.DataFrame = pd.read_csv(path)

        df = df[SubCol.raw_cols()]
        df = df.dropna(how="any")

        # Date type
        df[SubCol.Date] = df[SubCol.Date].astype("datetime64[ns]")

        # Convert GBP to USD
        gbp_usd: pd.Series = Ticker(ticker=YfTickers.GBPUSD, is_fx=True).prices
        for idx in df.index:
            row = df.loc[idx]

            date = row[SubCol.Date]
            df.loc[idx, SubCol.USD] = row[SubCol.GBP] * gbp_usd.loc[date]

        df = df.groupby(by=[SubCol.Date, SubCol.Portfolio]).sum()
        df = df.sort_values(by=SubCol.Date)
        df = df.reset_index().set_index(SubCol.Date, drop=True)
        df = df.resample("D").sum().fillna(0.0)

        return cls(data=df)
