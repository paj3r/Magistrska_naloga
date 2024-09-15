import pandas_datareader as pdr
import pandas as pd
import yfinance as yf
from datetime import datetime

def get_and_save_1day_data(names, startdate, enddate, filename):
    df = yf.download(names, start=startdate, end=enddate)
    df.sort_index(inplace=True)
    df["High"] = df["High"].astype(float).round(2)
    df["Open"] = df["Open"].astype(float).round(2)
    df["Low"] = df["Low"].astype(float).round(2)
    df["Close"] = df["Close"].astype(float).round(2)
    df["Adj Close"] = df["Adj Close"].astype(float).round(2)
    df["Volume"] = df["Volume"].astype(float).round(0)
    df.to_csv(filename)

if __name__ == '__main__':
    yf.pdr_override()
    get_and_save_1day_data("BTC-USD", "2018-01-01", "2024-06-30", "OHLCTestData/btcusd_ohlc_test.csv")
    get_and_save_1day_data("^GSPC", "2018-01-01", "2024-06-30", "OHLCTestData/snp500_ohlc_test.csv")
    get_and_save_1day_data("EURUSD=X", "2018-01-01", "2024-06-30", "OHLCTestData/eurusd_ohlc_test.csv")
