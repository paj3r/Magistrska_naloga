from backtesting import Strategy
from backtesting.lib import crossover, resample_apply
from backtesting import Backtest
from backtesting.test import GOOG, EURUSD
import pandas_ta as pta
import pandas as pd
import numpy as np
from ts2vg import NaturalVG
import talib
from visibility_graph import generate_new_node_naive

def VGCrossover_predictions(prices, positive):
    out = np.zeros(len(prices))
    if not positive:
        prices = [-x for x in prices]
    out[0] = 0
    for i in range(1, len(prices)):
        temp_prices = prices[:i]
        graph = NaturalVG()
        graph.build(temp_prices)

        matrix = graph.as_networkx()
        out[i] = matrix.number_of_edges()/matrix.number_of_nodes()
    return out

class VGCrossoverStrategy(Strategy):
    # Define the two MA lengths, for the AMA' calculation.
    # Used for later optimisation of these values.
    behind = 10
    def init(self):
        # Precompute the AMA' values.
        # self.rsi_window = 30
        # self.rsi = self.I(talib.RSI, self.data.Close, self.rsi_window)

        self.vg_signals = self.I(VGCrossover_predictions, self.data.Close, True)
        self.vg_signals_neg = self.I(VGCrossover_predictions, self.data.Close, False)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 10)

    def next(self):
        price = self.data.Close[-1]
        # If the price is larger than the calculated AMA', then send a buy signal, otherwise close.
        # if crossover(self.vg_signals, self.vg_signals_neg):
        #     self.buy()
        # else:
        #     self.sell()
        if not self.position and len(self.vg_signals) > self.behind:
            if self.vg_signals[-self.behind] < self.vg_signals[-1] and self.vg_signals[-self.behind] < self.vg_signals[-5]:
                self.buy(sl = price - self.atr[-1]*1, tp = price + self.atr[-1]*3)


dataframe = pd.read_csv("../OHLCTestData/btcusd_bull_ohlc.csv", parse_dates=True)
bt = Backtest(dataframe, VGCrossoverStrategy, cash=100000, exclusive_orders=True)
# stats = bt.optimize(behind=range(2, 50, 1), maximize='Return [%]')
# stats = bt.optimize(
#     short_length=range(5, 20, 5),
#     far_length=range(10, 100, 10),
#     maximize='Sharpe Ratio',
#     constraint=lambda param: param.short_length < param.far_length)
stats = bt.run()
print(stats)
bt.plot()
