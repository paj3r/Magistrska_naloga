import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import scipy
import pandas_datareader.data as pdr

from ts2vg import HorizontalVG, NaturalVG
from ts_to_vg import plot_ts_visibility # Plotting function for visibility graph


def get_1day_data(names, startdate, enddate):
    df = pdr.get_data_yahoo(names, startdate, enddate)['Close']
    df.sort_index(inplace=True)
    return df

yf.pdr_override()
dat = get_1day_data(["ETH-USD"], dt.datetime(2020, 1, 1), dt.datetime.now()).to_numpy()
ng = NaturalVG()
ng.build(dat)

adj_matrix = ng.adjacency_matrix()
#print(adj_matrix)

plot_ts_visibility(adj_matrix, dat)