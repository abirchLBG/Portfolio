from dataclasses import dataclass

import pandas as pd
from enum import StrEnum


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

    @classmethod
    def raw_cols(cls) -> list["TxCol"]:
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

    @classmethod
    def from_csv(cls, path: str) -> "Transactions":
        df: pd.DataFrame = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        df = df[TxCol.raw_cols()]

        # TODO: divs and cash balances not take into account
        # Drops dividends tx
        df = df.dropna(how="any")

        # Date type
        df[TxCol.Date] = df[TxCol.Date].astype("datetime64[ns]")

        # Add ticker col
        df[TxCol.Ticker] = df[TxCol.Description].apply(lambda x: DESC_TO_TICKER_MAP[x])

        # Change sign on SELL orders
        df.loc[df[TxCol.Transaction] == "Sale", TxCol.Quantity] = (
            df[TxCol.Quantity] * -1
        )

        # Format price
        df[TxCol.Price] = df[TxCol.Price].str.removeprefix("£").astype(float)

        # TODO: cash balances not taken into account
        # # Add Cash ticker to handle cash balance from SELL orders
        # gbp_usd: pd.Series = Ticker(ticker="GBPUSD=X", is_fx=True).prices

        # sell_tx: pd.DataFrame = df.loc[df[TxCol.Transaction] == "Sale"]

        # sell_tx[TxCol.Transaction] = "Purchase"
        # sell_tx[TxCol.Description] = "Cash"
        # sell_tx[TxCol.Ticker] = "USD"

        # # sell_tx[TxCol.Quantity] = sell_tx[TxCol.Price]

        # for i in sell_tx.index:
        #     tx_date: pd.Timestamp = sell_tx.loc[i, TxCol.Date] # type: ignore
        #     sell_tx.loc[i, TxCol.Quantity] = sell_tx.loc[i, TxCol.Quantity] * gbp_usd.loc[tx_date] * sell_tx.loc[i, TxCol.Price]

        # sell_tx[TxCol.Price] = 1.0
        # df = pd.concat([df, sell_tx], axis=0)

        df = df.sort_values(by=TxCol.Date).reset_index(drop=True)

        return cls(raw_data=df)

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
            .fillna(0)
        )
