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
import operator
from visibility_graph import generate_new_node_naive


def VG_center_point_prediction(close: np.array, lookback: int, positive:bool):
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
        pos_temp = nx.eccentricity(pos)
        neg_temp= nx.eccentricity(neg)
        pos_min = min(zip(pos_temp.values(), pos_temp.keys()))[0]
        neg_min = min(zip(neg_temp.values(), neg_temp.keys()))[0]
        pos_rez = [k for k, v in pos_temp.items() if v == pos_min]
        neg_rez = [k for k, v in neg_temp.items() if v == neg_min]

        # avg_short_dist_p[i] = all([dat[-(lookback - x)] < dat[-1] for x in pos_rez])
        # avg_short_dist_n[i] = all([dat[-(lookback - x)] > dat[-1] for x in neg_rez])
        avg_short_dist_p[i] = all([dat[-(lookback - x)] < dat[-1] for x in pos_rez])
        avg_short_dist_n[i] = all([dat[-(lookback - x)] > dat[-1] for x in neg_rez])

        # Another possibility...

    return avg_short_dist_p if positive else avg_short_dist_n



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

class VGCrossoverLongStrategyTesttwo(Strategy):
    # Define the two MA lengths, for the AMA' calculation.
    # Used for later optimisation of these values.
    ma_short_len = 10
    ma_long_len = 50
    lookback = 40
    def init(self):
        self.center_point = self.I(VG_center_point_prediction, self.data.Close, self.lookback, True)
        self.center_point_neg = self.I(VG_center_point_prediction, self.data.Close, self.lookback, False)
        self.short_ma = self.I(talib.MA, self.data.Close, self.ma_short_len)
        self.long_ma = self.I(talib.MA, self.data.Close, self.ma_long_len)
        # self.linreg = self.I(talib.LINEARREG, self.data.Close, self.ma_long_len)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 10)

    def next(self):
        price = self.data.Close[-1]
        if self.short_ma[-1] < self.long_ma[-1]:
            # Bear regime.
            if self.position.is_long:
                self.position.close()
            if not self.position and self.center_point_neg[-1]:
                self.sell(sl=price + self.atr[-1]*2)
            # if crossover(self.vg_spl_signals, self.vg_spl_signals_neg) or crossover(self.vg_signals, self.vg_signals_neg):
            #     self.position.close()
            # elif not self.position.is_short:
            #     self.sell()
        else:
            # Bull regime.
            if self.position.is_short:
                self.position.close()
            # if not self.position:
            #     self.buy(sl=price - self.atr[-1] * 2)
            if not self.position and self.center_point[-1]:
                self.buy(sl=price - self.atr[-1]*2)


dataframe = pd.read_csv("../OHLCTestData/btcusd_ohlc_test.csv", index_col=0, parse_dates=True, infer_datetime_format=True)
bt = Backtest(dataframe, VGCrossoverLongStrategyTest, cash=100000, exclusive_orders=True)
# stats = bt.optimize(lookback=range(10, 100, 10), maximize='Return [%]')
# stats = bt.optimize(
#     short_length=range(5, 20, 5),
#     far_length=range(10, 100, 10),
#     maximize='Sharpe Ratio',
#     constraint=lambda param: param.short_length < param.far_length)
stats = bt.run()
print(stats)
bt.plot()
