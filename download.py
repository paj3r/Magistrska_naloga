import pandas_datareader as pdr
import pandas as pd
import yfinance as yf
from datetime import datetime

def get_and_save_1day_data(names, startdate, enddate, filename):
    df = yf.download(names, start=startdate, end=enddate)
    df.sort_index(inplace=True)
    df.to_csv(filename)

if __name__ == '__main__':
    yf.pdr_override()
    get_and_save_1day_data("BTC-USD", "2018-01-01", "2023-12-31", "TestData/btcusd_ohlc.csv")
    get_and_save_1day_data("^GSPC", "2018-01-01", "2023-12-31", "TestData/snp500_ohlc.csv")
    get_and_save_1day_data("EURUSD=X", "2018-01-01", "2023-12-31", "TestData/eurusd_ohlc.csv")