from backtesting import Strategy
from backtesting.lib import crossover
from backtesting import Backtest
from backtesting.test import GOOG, EURUSD
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

    short_length = 10
    far_length = 20

    def init(self):
        # Precompute the AMA' values.
        self.daily_ama = self.I(calculate_AMA, self.data.Close, self.short_length, self.far_length)
        # self.short_ma = self.I(pta.sma, self.data.Close, 5)
        # self.long_ma = self.I(pta.sma, self.data.Close, 10)

    def next(self):
        # If the price is larger than the calculated AMA', then send a buy signal, otherwise close.
        if crossover(self.data.Close, self.daily_ama):
            self.position.close()
            self.buy()

        if crossover(self.daily_ama, self.data.Close):
            self.position.close()
            self.sell()



bt = Backtest(EURUSD, AmaStrategy, cash=10_000)
# stats = bt.optimize(
#     short_length=range(5, 20, 5),
#     far_length=range(10, 100, 10),
#     maximize='Sharpe Ratio',
#     constraint=lambda param: param.short_length < param.far_length)
stats = bt.run(short_length=5, far_length=10)
print(stats)
bt.plot()
