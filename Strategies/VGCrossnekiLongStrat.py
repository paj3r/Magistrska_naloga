from backtesting import Strategy
from backtesting.lib import crossover, resample_apply
from backtesting import Backtest
from backtesting.test import GOOG, EURUSD
import pandas_ta as pta
import pandas as pd
import numpy as np
from ts2vg import NaturalVG
import talib
import networkx as nx
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


def shortest_path_length(close: np.array, lookback: int, positive:bool):
    avg_short_dist_p = np.zeros(len(close))
    avg_short_dist_n = np.zeros(len(close))

    avg_short_dist_p[:] = np.nan
    avg_short_dist_n[:] = np.nan

    for i in range(lookback, len(close)):
        dat = close[i - lookback + 1: i + 1]

        pos = NaturalVG()
        pos.build(dat)

        neg = NaturalVG()
        neg.build(-dat)

        neg = neg.as_networkx()
        pos = pos.as_networkx()

        # you could replace shortest_path_length with other networkx metrics..
        avg_short_dist_p[i] = nx.average_shortest_path_length(pos)
        avg_short_dist_n[i] = nx.average_shortest_path_length(neg)

        # Another possibility...

    return avg_short_dist_p if positive else avg_short_dist_n

class VGCrossoverLongStrategy(Strategy):
    # Define the two MA lengths, for the AMA' calculation.
    # Used for later optimisation of these values.
    ma_short_len = 10
    ma_long_len = 50
    lookback = 20
    def init(self):
        self.vg_signals = self.I(VGCrossover_predictions, self.data.Close, True)
        self.vg_signals_neg = self.I(VGCrossover_predictions, self.data.Close, False)
        self.vg_spl_signals = self.I(shortest_path_length, self.data.Close, self.lookback, True)
        self.vg_spl_signals_neg = self.I(shortest_path_length, self.data.Close, self.lookback,  False)
        self.short_ma = self.I(talib.MA, self.data.Close, self.ma_short_len)
        self.long_ma = self.I(talib.MA, self.data.Close, self.ma_long_len)
        self.linreg = self.I(talib.LINEARREG, self.data.Close, self.ma_long_len)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 10)

    def next(self):
        price = self.data.Close[-1]
        if self.short_ma[-1] < self.long_ma[-1]:
            # Bear regime.
            if self.position.is_long:
                self.position.close()
            # if crossover(self.vg_spl_signals, self.vg_spl_signals_neg) or crossover(self.vg_signals, self.vg_signals_neg):
            #     self.position.close()
            # elif not self.position.is_short:
            #     self.sell()
        else:
            # Bull regime.
            if self.position.is_short:
                self.position.close()
            if not self.position and (crossover(self.vg_spl_signals, self.vg_spl_signals_neg) or crossover(self.vg_signals, self.vg_signals_neg)) and self.linreg[-1] > self.linreg[-2]:
                self.buy(sl = price - self.atr[-1]*2)


dataframe = pd.read_csv("../OHLCTestData/btcusd_ohlc.csv", index_col=0, parse_dates=True, infer_datetime_format=True)
bt = Backtest(dataframe, VGCrossoverLongStrategy, cash=100000, exclusive_orders=True)
# stats = bt.optimize(behind=range(2, 50, 1), maximize='Return [%]')
# stats = bt.optimize(
#     short_length=range(5, 20, 5),
#     far_length=range(10, 100, 10),
#     maximize='Sharpe Ratio',
#     constraint=lambda param: param.short_length < param.far_length)
stats = bt.run()
print(stats)
bt.plot()
