from dataclasses import dataclass

from src.dataclasses.transactions import Transactions


@dataclass
class Portfolio:
    transactions: Transactions
