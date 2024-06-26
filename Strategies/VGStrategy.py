from backtesting import Strategy
from backtesting.lib import crossover
from backtesting import Backtest
from backtesting.test import GOOG, EURUSD
import pandas_ta as pta
import pandas as pd
import numpy as np
from visibility_graph import generate_new_node_naive

def VG_predictions(prices, neighbourhood_size, positive):
    out = np.zeros(len(prices))
    if not positive:
        prices = [-x for x in prices]
    for i in range(neighbourhood_size, len(out)):
        temp_prices = prices[:i]
        out[i] = generate_new_node_naive(temp_prices, neighbourhood_size)
    if not positive:
        out = [-x for x in out]
    return out

class VGStrategy(Strategy):
    # Define the two MA lengths, for the AMA' calculation.
    # Used for later optimisation of these values.
    neighbourhood_size = 5

    def init(self):
        # Precompute the AMA' values.
        self.vg_signals = self.I(VG_predictions, self.data.Close, self.neighbourhood_size, True)
        self.vg_signals_neg = self.I(VG_predictions, self.data.Close, self.neighbourhood_size, False)
        # self.short_ma = self.I(pta.sma, self.data.Close, 5)
        # self.long_ma = self.I(pta.sma, self.data.Close, 10)

    def next(self):
        # If the price is larger than the calculated AMA', then send a buy signal, otherwise close.
        mid = (self.vg_signals[-1] + self.vg_signals_neg[-1])/2
        if mid > self.data["Close"][-1]:
            self.buy()
        else:
            self.sell()


dataframe = pd.read_csv("../OHLCTestData/btcusd_ohlc.csv", parse_dates=True)
bt = Backtest(dataframe, VGStrategy, cash=100000, commission=.002, exclusive_orders=True)
#stats = bt.optimize(neighbourhood_size=range(2, 15, 1), maximize='Return [%]')
# stats = bt.optimize(
#     short_length=range(5, 20, 5),
#     far_length=range(10, 100, 10),
#     maximize='Sharpe Ratio',
#     constraint=lambda param: param.short_length < param.far_length)
stats = bt.run(neighbourhood_size=5)
print(stats)
bt.plot()
