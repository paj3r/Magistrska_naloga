import datetime as dt

import networkx as nx
import pandas_datareader.data as pdr
import yfinance as yf
from ts2vg import NaturalVG, HorizontalVG
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ts_to_vg import plot_ts_visibility  # Plotting function for visibility graph


def vizualize_matrix(matrix, data ,positive:bool):
    n_nodes = len(matrix.nodes)
    posx = list(range(n_nodes))
    posy = data
    pos = {i: [posx[i], posy[i]] for i in range(n_nodes)}
    nx.draw(matrix, pos=pos, connectionstyle=f'arc3,rad={-0.7 if positive else 0.7}', arrows=True)
    plt.ylim([min(data)-1, max(data)+2])
    plt.show()

def generate_new_node(time_series_values, matrix:nx.classes.graph.Graph, neg_matrix:nx.classes.graph.Graph, neighbourhood_size: int):
    n_nodes = len(matrix.nodes)
    #vizualize_matrix(matrix, time_series_values, True)
    #vizualize_matrix(neg_matrix, time_series_values, False)

    #create new node
    neighbourhood = matrix.subgraph(list(range(n_nodes))[-neighbourhood_size:]).copy()
    nx.draw(neighbourhood, with_labels=True)
    plt.show()
    matrix.add_node(n_nodes)
    neighbourhood.add_node(n_nodes)
    neighbourhood.add_edge(n_nodes-1, n_nodes)
    sim = nx.panther_similarity(neighbourhood, n_nodes)
    nx.draw(neighbourhood, with_labels=True)
    plt.show()
    pass




def get_1day_data(names, startdate, enddate):
    df = pdr.get_data_yahoo(names, startdate, enddate)['Close']
    df.sort_index(inplace=True)
    return df


yf.pdr_override()
# dat = get_1day_data(["ETH-USD"], dt.datetime(2020, 1, 1), dt.datetime.now()).to_numpy()
dat = [1.0, 0.5, 0.3, 0.7, 1.0, 0.5, 0.3, 0.8]
neg_dat = [-x for x in dat]
graph = NaturalVG()
graph.build(dat)
neg_graph = NaturalVG()
neg_graph.build(neg_dat)

network = graph.as_networkx()
neg_network = neg_graph.as_networkx()

generate_new_node(dat, network, neg_network, 4)
# print(adj_matrix)

plot_ts_visibility(adj_matrix, dat)
