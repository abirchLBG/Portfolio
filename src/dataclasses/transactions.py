from dataclasses import dataclass
import logging
from enum import StrEnum

import pandas as pd

logger = logging.getLogger(__name__)


DESC_TO_TICKER_MAP: dict[str, str] = {
    "Meta Platforms Inc Class A": "META",
    "SPDR S&P US Technology Select Sect ETF GBP": "GXLK.L",
    "Amazon.com Inc": "AMZN",
    "Apple Inc": "AAPL",
    "Alphabet Inc Class C": "GOOG",
    "Oracle Corp": "ORCL",
    "iShares Core S&P 500 ETF USD Acc GBP": "CSP1.L",
    "Amundi DJ Industrial Average ETF Dist GBP": "DJEL.L",
    "Micron Technology Inc": "MU",
    "GameStop Corp Class A": "GME",
    "Vanguard S&P 500 UCITS ETF GBP": "VUSA.L",
}


class TxCol(StrEnum):
    # Raw
    Portfolio = "Portfolio"
    Date = "Date"
    Transaction = "Transaction"
    Description = "Description"
    Quantity = "Quantity"
    Price = "Price"

    # Derived
    Ticker = "Ticker"

    @staticmethod
    def raw_cols() -> list["TxCol"]:
        return [
            TxCol.Portfolio,
            TxCol.Date,
            TxCol.Transaction,
            TxCol.Description,
            TxCol.Quantity,
            TxCol.Price,
        ]


@dataclass
class Transactions:
    raw_data: pd.DataFrame

    @staticmethod
    def from_csv(path: str, ticker_map: dict[str, str] | None = None) -> "Transactions":
        """
        Load transaction data from CSV file.

        Args:
            path: Path to CSV file
            ticker_map: Optional custom mapping from description to ticker symbol.
                       If None, uses default DESC_TO_TICKER_MAP.

        Returns:
            Transactions object

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If CSV has invalid structure or data
            KeyError: If description not found in ticker map
        """
        ticker_map = ticker_map or DESC_TO_TICKER_MAP

        try:
            df: pd.DataFrame = pd.read_csv(path)
        except FileNotFoundError:
            logger.error(f"Transactions CSV file not found: {path}")
            raise
        except Exception as e:
            logger.error(f"Failed to read transactions CSV from {path}: {e}")
            raise ValueError(f"Failed to read transactions CSV: {e}") from e

        try:
            df.columns = df.columns.str.strip()
        except Exception as e:
            logger.error(f"Failed to process CSV columns: {e}")
            raise ValueError(f"Invalid CSV structure: {e}") from e

        # Validate required columns
        required_cols = TxCol.raw_cols()
        missing_cols = set(required_cols) - set(df.columns)  # type: ignore
        if missing_cols:
            raise ValueError(
                f"Transactions CSV missing required columns: {missing_cols}. "
                f"Expected: {required_cols}"
            )

        df = df[TxCol.raw_cols()]

        # TODO: divs and cash balances not taken into account
        # Drops dividends tx
        df = df.dropna(how="any")

        if df.empty:
            logger.warning(f"Transactions CSV at {path} is empty after removing NaNs")
            return Transactions(raw_data=pd.DataFrame())

        # Date type conversion
        try:
            df[TxCol.Date] = pd.to_datetime(df[TxCol.Date], format="%d/%m/%Y")
        except Exception as e:
            logger.error(f"Failed to parse dates in transactions CSV: {e}")
            raise ValueError(f"Invalid date format in transactions CSV: {e}") from e

        # Add ticker col with error handling
        def map_description_to_ticker(desc: str) -> str:
            if desc not in ticker_map:
                logger.error(f"Unknown security description: {desc}")
                raise KeyError(
                    f"Description '{desc}' not found in ticker map. "
                    f"Available: {list(ticker_map.keys())}"
                )
            return ticker_map[desc]

        try:
            df[TxCol.Ticker] = df[TxCol.Description].apply(map_description_to_ticker)
        except KeyError:
            raise

        # Change sign on SELL orders
        try:
            df.loc[df[TxCol.Transaction] == "Sale", TxCol.Quantity] = (
                df[TxCol.Quantity] * -1
            )
        except Exception as e:
            logger.error(f"Failed to process transaction types: {e}")
            raise ValueError(f"Invalid transaction data: {e}") from e

        # Format price - handle different currency symbols
        try:
            df[TxCol.Price] = (
                df[TxCol.Price].str.replace(r"[£$€]", "", regex=True).astype(float)
            )
        except Exception as e:
            logger.error(f"Failed to parse prices: {e}")
            raise ValueError(f"Invalid price format in transactions CSV: {e}") from e

        try:
            df = df.sort_values(by=TxCol.Date).reset_index(drop=True)
        except Exception as e:
            logger.error(f"Failed to sort transaction data: {e}")
            raise ValueError(f"Data processing failed: {e}") from e

        return Transactions(raw_data=df)

    def __post_init__(self) -> None:
        self.data: pd.DataFrame = (
            self.raw_data.pivot_table(
                index=TxCol.Date,
                columns=TxCol.Ticker,
                values=TxCol.Quantity,
                aggfunc="sum",
            )
            .resample("D")
            .sum()
        )

        self.data = self.data.reindex(
            pd.DatetimeIndex(
                pd.date_range(
                    self.data.index.min(), pd.Timestamp.now().date(), freq="D"
                )
            )
        ).fillna(0)
