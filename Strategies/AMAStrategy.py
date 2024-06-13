from backtesting import Strategy
from backtesting.lib import crossover
from backtesting import Backtest
from backtesting.test import GOOG, EURUSD
import pandas as pd
import pandas_ta as pta
import numpy as np


def calculate_AMA(prices, short, far):
    out = np.zeros(len(prices))
    for i in range(far, len(out)):
        temp_prices = prices[:i]
        a = np.std(temp_prices[-short:]).item()
        if a == 0:
            continue
        b = np.std(temp_prices[-far:]).item()
        v = b / a + short
        p = int(round(v))
        cut_pric = temp_prices[-p:]
        k = np.sum(cut_pric).item()
        ama = k / v
        out[i] = ama
    out[out == 0] = np.nan
    return out


class AmaStrategy(Strategy):
    # Define the two MA lengths, for the AMA' calculation.
    # Used for later optimisation of these values.

    short_length = 5
    far_length = 10

    def init(self):
        # Precompute the AMA' values.
        self.daily_ama = self.I(calculate_AMA, self.data.Close, self.short_length, self.far_length)
        # self.short_ma = self.I(pta.sma, self.data.Close, 5)
        # self.long_ma = self.I(pta.sma, self.data.Close, 10)

    def next(self):
        # If the price is larger than the calculated AMA', then send a buy signal, otherwise close.
        if crossover(self.data.Close, self.daily_ama) and not self.position.is_long:
            self.buy()
        elif crossover(self.daily_ama, self.data.Close):
            self.position.close()



dataframe = pd.read_csv("../OHLCTestData/btcusd_ohlc.csv", parse_dates=True)
bt = Backtest(dataframe, AmaStrategy, cash=100000, exclusive_orders=True)
stats = bt.run(short_length=5, far_length=10)
print(stats)
bt.plot()
