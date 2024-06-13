from backtesting import Strategy
from backtesting.lib import crossover
import pandas as pd
from backtesting import Backtest
from backtesting.test import GOOG, EURUSD
import pandas_ta as pta
import numpy as np
from visibility_graph import generate_new_node_zhang

def VG_Zhang_predictions(prices, positive):
    out = np.zeros(len(prices))
    if not positive:
        prices = [-x for x in prices]
    for i in range(1, len(out)):
        temp_prices = prices[:i]
        out[i] = generate_new_node_zhang(temp_prices)
    if not positive:
        out = [-x for x in out]
    return out

class VGZhangStrategy(Strategy):
    # Define the two MA lengths, for the AMA' calculation.
    # Used for later optimisation of these values.

    def init(self):
        # Precompute the AMA' values.
        self.zhang_signals = self.I(VG_Zhang_predictions, self.data.Close, True)
        # self.vg_signals_neg = self.I(VG_predictions, self.data.Close, self.neighbourhood_size, False)
        # self.short_ma = self.I(pta.sma, self.data.Close, 5)
        # self.long_ma = self.I(pta.sma, self.data.Close, 10)

    def next(self):
        # If the price is larger than the calculated AMA', then send a buy signal, otherwise close.
        mid = (self.zhang_signals[-1])
        if mid > self.data["Close"][-1]:
            self.buy()
        else:
            self.sell()


dataframe = pd.read_csv("../OHLCTestData/btcusd_ohlc.csv", parse_dates=True)
bt = Backtest(dataframe, VGZhangStrategy, cash=100000,  commission=.002, exclusive_orders=True)
#stats = bt.optimize(neighbourhood_size=range(2, 15, 1), maximize='Return [%]')
# stats = bt.optimize(
#     short_length=range(5, 20, 5),
#     far_length=range(10, 100, 10),
#     maximize='Sharpe Ratio',
#     constraint=lambda param: param.short_length < param.far_length)
stats = bt.run()
print(stats)
bt.plot()
