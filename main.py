import datetime as dt

import networkx as nx
import pandas_datareader.data as pdr
import yfinance as yf
from ts2vg import NaturalVG, HorizontalVG
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ts_to_vg import plot_ts_visibility  # Plotting function for visibility graph


def vizualize_matrix(matrix, data, positive: bool):
    n_nodes = len(matrix.nodes)
    posx = list(range(n_nodes))
    posy = data
    pos = {i: [posx[i], posy[i]] for i in range(n_nodes)}
    nx.draw(matrix, pos=pos,
            #connectionstyle=f'arc3,rad={-0.7 if positive else 0.7}',
            arrows=True, with_labels=True)
    plt.ylim([min(data) - 1, max(data) + 2])
    plt.show()


def neighbourhood_simmilarity(matrix, neighbourhood, target_node):
    sum = 0
    for node in neighbourhood.nodes:
        sum += nx.simrank_similarity(matrix, node, target_node)
    return sum / len(neighbourhood)


def generate_new_node(time_series_values, matrix: nx.classes.graph.Graph, neighbourhood_size: int, positive: bool):
    new_node_index = len(matrix.nodes)

    # Create new node.
    neighbourhood = matrix.subgraph(list(range(new_node_index))[-neighbourhood_size:]).copy()
    matrix.add_node(new_node_index)
    matrix.add_edge(new_node_index - 1, new_node_index)

    # Add nodes.
    for node in neighbourhood.nodes:
        previous_sim = -1
        node_edges = list(matrix.edges(node))
        for edge in node_edges:
            matrix.add_edge(new_node_index, edge[1])
            sim = neighbourhood_simmilarity(matrix, neighbourhood, new_node_index)
            if sim <= previous_sim:
                matrix.remove_edge(new_node_index, edge[1])
                sim = neighbourhood_simmilarity(matrix, neighbourhood, new_node_index)
            previous_sim = sim

    # Remove nodes.
    previous_sim = neighbourhood_simmilarity(matrix, neighbourhood, new_node_index)
    new_edges = list(matrix.edges(new_node_index))
    for edge in new_edges:
        matrix.remove_edge(new_node_index, edge[1])
        sim = neighbourhood_simmilarity(matrix, neighbourhood, new_node_index)
        if sim <= previous_sim:
            matrix.add_edge(new_node_index, edge[1])
            sim = neighbourhood_simmilarity(matrix, neighbourhood, new_node_index)
        previous_sim = sim

    # In visibility graph, neighbouring nodes should always be connected.
    if not matrix.has_edge(new_node_index-1, new_node_index):
        matrix.add_edge(new_node_index, new_node_index-1)

    # A node should not be connected to itself.
    if matrix.has_edge(new_node_index, new_node_index):
        matrix.remove_edge(new_node_index, new_node_index)

    new_value = 0

    connected_edges = [x[1] for x in list(matrix.edges(new_node_index))]
    connected_edges.sort()
    local_values = [time_series_values[x] for x in connected_edges]
    local_maximum = max(local_values)
    local_minimum = min(local_values)

    value = local_minimum
    for index in range(new_node_index-1, new_node_index-neighbourhood_size-1, -1):
        pass
        #do logic

    time_series_values.append(local_maximum)

    vizualize_matrix(matrix, time_series_values, positive)
    #nx.draw(matrix, with_labels=True)
    #plt.show()


def get_1day_data(names, startdate, enddate):
    df = pdr.get_data_yahoo(names, startdate, enddate)['Close']
    df.sort_index(inplace=True)
    return df


if __name__ == '__main__':
    yf.pdr_override()
    # dat = get_1day_data(["ETH-USD"], dt.datetime(2020, 1, 1), dt.datetime.now()).to_numpy()
    dat = [1.0, 0.5, 0.3, 0.7, 1.0, 0.5, 0.3, 0.8, 1.0, 0.4]
    neg_dat = [-x for x in dat]
    graph = NaturalVG()
    graph.build(dat)
    neg_graph = NaturalVG()
    neg_graph.build(neg_dat)

    network = graph.as_networkx()
    neg_network = neg_graph.as_networkx()

    generate_new_node(dat, network,  4, True)
    generate_new_node(dat, neg_network, 4, False)
    # print(adj_matrix)

    pass
