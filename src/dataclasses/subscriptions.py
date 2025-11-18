from dataclasses import dataclass
from enum import StrEnum
import logging

import pandas as pd

from src.constants import YfTickers
from src.dataclasses.ticker import Ticker, TickerDataError

logger = logging.getLogger(__name__)


class SubCol(StrEnum):
    # Raw
    Portfolio = "Portfolio"
    Date = "Date"
    GBP = "GBP"

    # Derived
    USD = "USD"

    @staticmethod
    def raw_cols() -> list["SubCol"]:
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

    @staticmethod
    def from_csv(path: str) -> "Subscriptions":
        """
        Load subscription data from CSV file.

        Args:
            path: Path to CSV file

        Returns:
            Subscriptions object

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If CSV has invalid structure or data
            TickerDataError: If FX data cannot be fetched
        """
        try:
            df: pd.DataFrame = pd.read_csv(path)
        except FileNotFoundError:
            logger.error(f"Subscriptions CSV file not found: {path}")
            raise
        except Exception as e:
            logger.error(f"Failed to read subscriptions CSV from {path}: {e}")
            raise ValueError(f"Failed to read subscriptions CSV: {e}") from e

        # Validate required columns
        required_cols = SubCol.raw_cols()
        missing_cols = set(required_cols) - set(df.columns)  # type: ignore
        if missing_cols:
            raise ValueError(
                f"Subscriptions CSV missing required columns: {missing_cols}. "
                f"Expected: {required_cols}"
            )

        df = df[SubCol.raw_cols()]
        df = df.dropna(how="any")

        if df.empty:
            logger.warning(f"Subscriptions CSV at {path} is empty after removing NaNs")
            return Subscriptions(data=pd.DataFrame())

        # Date type conversion
        try:
            df[SubCol.Date] = pd.to_datetime(df[SubCol.Date], format="%d/%m/%Y")
        except Exception as e:
            logger.error(f"Failed to parse dates in subscriptions CSV: {e}")
            raise ValueError(f"Invalid date format in subscriptions CSV: {e}") from e

        # Convert GBP to USD
        try:
            gbp_usd: pd.Series = Ticker(ticker=YfTickers.GBPUSD, is_fx=True).prices
        except TickerDataError as e:
            logger.error(f"Failed to fetch GBP/USD exchange rate: {e}")
            raise

        try:
            for idx in df.index:
                row = df.loc[idx]
                date = row[SubCol.Date]

                # Check if date exists in FX data
                if date not in gbp_usd.index:
                    # Find nearest date
                    nearest_date = gbp_usd.index[
                        gbp_usd.index.get_indexer([date], method="nearest")[0]
                    ]
                    logger.debug(
                        f"Date {date} not in FX data, using nearest date {nearest_date}"
                    )
                    fx_rate = gbp_usd.loc[nearest_date]
                else:
                    fx_rate = gbp_usd.loc[date]

                df.loc[idx, SubCol.USD] = row[SubCol.GBP] * fx_rate

        except Exception as e:
            logger.error(f"Failed to convert GBP to USD: {e}")
            raise ValueError(f"Currency conversion failed: {e}") from e

        try:
            df = df.groupby(by=[SubCol.Date, SubCol.Portfolio]).sum()
            df = df.sort_values(by=SubCol.Date)
            df = df.reset_index().set_index(SubCol.Date, drop=True)
            df = df.resample("D").sum().fillna(0.0)
        except Exception as e:
            logger.error(f"Failed to process subscription data: {e}")
            raise ValueError(f"Data processing failed: {e}") from e

        return Subscriptions(data=df)
