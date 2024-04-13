import pandas as pd
import yfinance as yf
import re
import numpy as np
from termcolor import colored
import sys


def get_currency(x):
    if pd.notna(x) and x != "":
        return yf.Ticker(x).basic_info["currency"]
    else:
        return ""


class Portfolio:
    def __init__(self, path=sys.path[0] + "/portfolio.csv",):
        self.path = path
        self.df_raw = pd.read_csv(path, usecols=["Investment","Quantity"], index_col=False)
        self.usd_gbp = yf.Ticker("USDGBP=X").history()["Close"][-1]
        self.cash = {"GBP": self.df_raw.iloc[-1]["Quantity"],
                     "USD": round(self.df_raw.iloc[-1]["Quantity"] * (1 / self.usd_gbp), 2)}
        self._init_stocks_df()

    def __repr__(self):
        return self.df_stocks.to_string()
    

    def _init_stocks_df(self):
        self.df_stocks = self.df_raw.copy().iloc[0:-1]

        self.df_stocks["Exchange"] = self.df_stocks["Investment"].apply(lambda x: re.search(r"\((.*?)\:", x).group()[1:-1] if re.search(r"\((.*?)\:", x) else "")
        self.df_stocks["Ticker"] = self.df_stocks["Investment"].apply(lambda x: re.search(r"\:(.*?)\)", x).group()[1:-1] if re.search(r"\((.*?)\:", x) else "")
        self.df_stocks["Ticker"] = self.df_stocks["Ticker"] + self.df_stocks["Exchange"].apply(lambda x: ".L" if x == "LSE" else "")
        
        self.df_stocks["Currency"] = ""
        self.df_stocks["Currency"] = self.df_stocks["Ticker"].where(self.df_stocks["Ticker"] != "").apply(get_currency)


        self.df_stocks["Price Raw"] = self.df_stocks["Ticker"].apply(lambda x: yf.Ticker(x).history()["Close"][-1]).round(2)
        
        self.df_stocks["Price GBP"] = np.nan
        self.df_stocks["Price GBP"] = self.df_stocks["Price GBP"].where(self.df_stocks["Currency"] != "GBP", self.df_stocks["Price Raw"]).round(2)
        self.df_stocks["Price GBP"] = self.df_stocks["Price GBP"].where(self.df_stocks["Currency"] != "USD", self.df_stocks["Price Raw"] * self.usd_gbp).round(2)
        self.df_stocks["Price GBP"] = self.df_stocks["Price GBP"].where(self.df_stocks["Currency"] != "GBp", self.df_stocks["Price Raw"] / 100).round(2)
        # self.df_stocks["Price USD"] = (self.df_stocks["Price GBP"] * (1 / self.usd_gbp)).round(2)

        
        self.df_stocks["Price GBP 1D"] = self.df_stocks["Ticker"].apply(lambda x: yf.Ticker(x).history()["Close"][-2])
        self.df_stocks["Price GBP 1D"] = self.df_stocks["Price GBP 1D"].where(self.df_stocks["Currency"] != "GBP", self.df_stocks["Price GBP 1D"]).round(2)
        self.df_stocks["Price GBP 1D"] = self.df_stocks["Price GBP 1D"].where(self.df_stocks["Currency"] != "USD", self.df_stocks["Price GBP 1D"] * self.usd_gbp).round(2)
        self.df_stocks["Price GBP 1D"] = self.df_stocks["Price GBP 1D"].where(self.df_stocks["Currency"] != "GBp", self.df_stocks["Price GBP 1D"] / 100).round(2)


    def show_portoflio_totals(self):
        self.total_gbp = (self.df_stocks["Price GBP"] * self.df_stocks["Quantity"]).sum() + self.cash['GBP']
        total_gbp_1d = ((self.df_stocks['Price GBP 1D'] * self.df_stocks["Quantity"]).sum() + self.cash['GBP']).round(2)
        pct_change = round(100 * (self.total_gbp - total_gbp_1d) / total_gbp_1d, 2)

        if self.total_gbp > total_gbp_1d:
            color = "light_green"
        else:
            color = "light_red"

        print(colored("Portfolio totals:", attrs=("underline", "bold", "blink")))
        print(colored(f"£ {self.total_gbp:,.2f} | {pct_change}%", color))
        print(colored(f"$ {self.total_gbp * 1/self.usd_gbp:,.2f} | {pct_change}%", color))


    def show_forecast(self, years):
        print()
        print(f"{years}Y forecast:")
        print(f"£ {round(self.total_gbp * 1.09**years, 2):,.2f}")
        print(f"$ {round(self.total_gbp * 1/self.usd_gbp * 1.09**years, 2):,.2f}")


if __name__ == "__main__":
    portfolio = Portfolio()
    portfolio._init_stocks_df()
    portfolio.show_portoflio_totals()
    # portfolio.show_forecast(10)
    # portfolio.show_forecast(20)
    # portfolio.show_forecast(30)
    # portfolio.show_forecast(40)
