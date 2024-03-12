import datetime as dt

import networkx as nx
import numpy as np
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
            # connectionstyle=f'arc3,rad={-0.7 if positive else 0.7}',
            arrows=True, with_labels=True)
    plt.ylim([min(data) - 1, max(data) + 2])
    plt.show()


def neighbourhood_simmilarity(matrix, neighbourhood, target_node):
    sum = 0
    if len(neighbourhood) == 0:
        return 0
    node_pairs = [(target_node, x) for x in neighbourhood.nodes]
    preds = nx.jaccard_coefficient(matrix, node_pairs)
    for u, v, p in preds:
        sum += p
    return sum / len(neighbourhood)


def generate_new_node_naive(time_series_values, matrix: nx.classes.graph.Graph, neighbourhood_size: int, positive: bool,
                            draw: bool):
    new_node_index = len(matrix.nodes)
    max_neighbourhood_size = neighbourhood_size*3

    # Create new node.
    neighbourhood = matrix.subgraph(list(range(new_node_index))[-neighbourhood_size:]).copy()
    matrix.add_node(new_node_index)
    matrix.add_edge(new_node_index - 1, new_node_index)

    # Add nodes.
    for node in neighbourhood.nodes:
        previous_sim = -1
        node_edges = list(matrix.edges(node))
        if len(node_edges) > max_neighbourhood_size:
            pass
        # Filter out long values
        for edge in node_edges:
            if edge[1] < new_node_index-max_neighbourhood_size:
                node_edges.remove(edge)

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
    if not matrix.has_edge(new_node_index - 1, new_node_index):
        matrix.add_edge(new_node_index, new_node_index - 1)

    # A node should not be connected to itself.
    if matrix.has_edge(new_node_index, new_node_index):
        matrix.remove_edge(new_node_index, new_node_index)

    predicted_value = 0

    connected_edges = [x[1] for x in list(matrix.edges(new_node_index))]
    connected_edges.sort()

    new_edges = matrix.edges(new_node_index)
    predicted_value = 0
    for edge in new_edges:
        predicted_value += time_series_values[edge[1]] / len(new_edges)

    time_series_values = np.append(time_series_values, predicted_value)

    if draw:
        vizualize_matrix(matrix, time_series_values, positive)

    return time_series_values[-1]


def get_1day_data(names, startdate, enddate):
    df = pdr.get_data_yahoo(names, startdate, enddate)['Close']
    df.sort_index(inplace=True)
    return df


if __name__ == '__main__':
    yf.pdr_override()
    dat = get_1day_data(["ETH-USD"], dt.datetime(2023, 6, 5), dt.datetime.now()).to_numpy()
    # dat = [1.0, 0.5, 0.3, 0.7, 1.0, 0.5, 0.3, 0.8, 1.0, 0.4]
    neighbourhood_range = range(2, 30)
    neg_dat = [-x for x in dat]
    graph = NaturalVG()
    graph.build(dat)
    neg_graph = NaturalVG()
    neg_graph.build(neg_dat)

    network = graph.as_networkx()
    neg_network = neg_graph.as_networkx()

    # new_value = generate_new_node_naive(dat, network, neighbourhood_size, True, False)
    # new_value_neg = generate_new_node_naive(dat, neg_network, neighbourhood_size, False, False)
    # print(adj_matrix)
    for neighbourhood_size in neighbourhood_range:
        print("Neighbourhood size:", neighbourhood_size)
        predictions = np.zeros(neighbourhood_size)
        neg_predictions = np.zeros(neighbourhood_size)
        for i in range(neighbourhood_size, len(dat)):
            rolling_dat = dat[:i]
            rolling_neg_dat = [-x for x in rolling_dat]
            rolling_network = network.subgraph(list(range(0, i))).copy()
            rolling_neg_network = neg_network.subgraph(list(range(0, i))).copy()
            new_value = generate_new_node_naive(rolling_dat, rolling_network, neighbourhood_size, True, False)
            new_value_neg = generate_new_node_naive(rolling_dat, rolling_neg_network, neighbourhood_size,
                                                    False, False)
            predictions = np.append(predictions, new_value)
            neg_predictions = np.append(neg_predictions, new_value_neg)

        diff = dat - predictions
        neg_diff = dat - neg_predictions
        avg_diff = (dat - (predictions + neg_predictions)/2)
        print("Avg diff on positive: ", np.average(diff))
        print("Avg diff on negative: ", np.average(neg_diff))
        print("Avg diff: ", np.average(avg_diff))
        print("\n")
        pass

