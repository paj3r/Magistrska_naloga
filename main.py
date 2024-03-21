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


def generate_new_node_zhang(time_series_values, matrix:nx.classes.graph.Graph, ahead=1):
    # proposed method in this paper: https://www.sciencedirect.com/science/article/pii/S0378437117310622
    max_sim = 0
    max_sim_ix = 0
    last_node = len(matrix.nodes)-1
    for node in matrix.nodes:
        if node == last_node:
            break
        preds = nx.jaccard_coefficient(matrix, [(node, last_node)])
        for u, v, p in preds:
            temp_sim = p
        if temp_sim > max_sim:
            max_sim = temp_sim
            max_sim_ix = node

    prediction = ((time_series_values[last_node] - time_series_values[max_sim_ix])/(last_node - max_sim_ix)
                  + time_series_values[last_node])
    return prediction


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
        # if len(node_edges) > max_neighbourhood_size:
        #     pass
        # # Filter out long values
        # for edge in node_edges:
        #     if edge[1] < new_node_index-max_neighbourhood_size:
        #         node_edges.remove(edge)

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

def test_naive(dat):
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
    max_pos = 0
    max_pos_ix = 0
    max_neg = 0
    max_neg_ix = 0
    max_avg = 0
    max_avg_ix = 0
    for neighbourhood_size in neighbourhood_range:
        print("Neighbourhood size:", neighbourhood_size)
        predictions = np.zeros(neighbourhood_size)
        neg_predictions = np.zeros(neighbourhood_size)
        for i in range(neighbourhood_size, len(dat)):
            rolling_dat = dat[:i]
            rolling_network = network.subgraph(list(range(0, i))).copy()
            rolling_neg_network = neg_network.subgraph(list(range(0, i))).copy()
            new_value = generate_new_node_naive(rolling_dat, rolling_network, neighbourhood_size,
                                                    True, False)
            new_value_neg = generate_new_node_naive(rolling_dat, rolling_neg_network, neighbourhood_size,
                                                    False, False)
            predictions = np.append(predictions, new_value)
            neg_predictions = np.append(neg_predictions, new_value_neg)

        # Check positive hitrate
        hits_pos = 0
        data = 0
        for i in range(1, len(dat)):
            if dat[i - 1] < predictions[i] and dat[i - 1] < dat[i]:
                hits_pos += 1
            elif dat[i - 1] > predictions[i] and dat[i - 1] > dat[i]:
                hits_pos += 1
            data += 1
        if max_pos < hits_pos / data:
            max_pos = hits_pos / data
            max_pos_ix = neighbourhood_size
        hits_neg = 0
        # Check negative hitrate
        for i in range(1, len(dat)):
            if dat[i - 1] < neg_predictions[i] and dat[i - 1] < dat[i]:
                hits_neg += 1
            elif dat[i - 1] > neg_predictions[i] and dat[i - 1] > dat[i]:
                hits_neg += 1
        if max_neg < hits_neg / data:
            max_neg = hits_neg / data
            max_neg_ix = neighbourhood_size

        hits_avg = 0
        # Check negative hitrate
        for i in range(1, len(dat)):
            val = (neg_predictions[i] + predictions[i]) / 2
            if dat[i - 1] < val and dat[i - 1] < dat[i]:
                hits_avg += 1
            elif dat[i - 1] > val and dat[i - 1] > dat[i]:
                hits_avg += 1
        if max_avg < hits_avg / data:
            max_avg = hits_avg / data
            max_avg_ix = neighbourhood_size
        print("Data length: ", data)
        print("Hits: ", hits_pos)
        print("Hitrate pos: " + str(hits_pos / data))
        print("Hits neg: ", hits_neg)
        print("Hitrate neg: " + str(hits_neg / data))
        print("Hits avg: ", hits_avg)
        print("Hitrate avg: " + str(hits_avg / data))
        # diff = dat - predictions
        # neg_diff = dat - neg_predictions
        # avg_diff = (dat - (predictions + neg_predictions)/2)
        # print("Avg diff on positive: ", np.average(diff))
        # print("Avg diff on negative: ", np.average(neg_diff))
        # print("Avg diff: ", np.average(avg_diff))
        print("\n")
        pass
    print("Max positive: ", max_pos, " At neighbourhood size:", max_pos_ix)
    print("Max negative: ", max_neg, " At neighbourhood size:", max_neg_ix)
    print("Max average: ", max_avg, " At neighbourhood size:", max_avg_ix)

def test_zhang(dat):
    neg_dat = [-x for x in dat]
    graph = NaturalVG()
    graph.build(dat)
    neg_graph = NaturalVG()
    neg_graph.build(neg_dat)
    network = graph.as_networkx()
    neg_network = neg_graph.as_networkx()

    hits = 0
    data = 0
    predictions = np.zeros(1)
    neg_predictions = np.zeros(1)
    for i in range(1, len(dat)):
        rolling_dat = dat[:i]
        rolling_network = network.subgraph(list(range(0, i))).copy()
        rolling_neg_network = neg_network.subgraph(list(range(0, i))).copy()
        new_value = generate_new_node_zhang(rolling_dat, rolling_network)
        new_value_neg = generate_new_node_zhang(rolling_dat, rolling_neg_network)
        predictions = np.append(predictions, new_value)
        neg_predictions = np.append(neg_predictions, new_value_neg)

    # Check positive hitrate
    hits_pos = 0
    data = 0
    for i in range(1, len(dat)):
        if dat[i - 1] < predictions[i] and dat[i - 1] < dat[i]:
            hits_pos += 1
        elif dat[i - 1] > predictions[i] and dat[i - 1] > dat[i]:
            hits_pos += 1
        data += 1

    hits_neg = 0
    # Check negative hitrate
    for i in range(1, len(dat)):
        if dat[i - 1] < neg_predictions[i] and dat[i - 1] < dat[i]:
            hits_neg += 1
        elif dat[i - 1] > neg_predictions[i] and dat[i - 1] > dat[i]:
            hits_neg += 1

    hits_avg = 0
    # Check negative hitrate
    for i in range(1, len(dat)):
        val = (neg_predictions[i] + predictions[i]) / 2
        if dat[i - 1] < val and dat[i - 1] < dat[i]:
            hits_avg += 1
        elif dat[i - 1] > val and dat[i - 1] > dat[i]:
            hits_avg += 1

    print("Data length: ", data)
    print("Hits: ", hits_pos)
    print("Hitrate pos: " + str(hits_pos / data))
    print("Hits neg: ", hits_neg)
    print("Hitrate neg: " + str(hits_neg / data))
    print("Hits avg: ", hits_avg)
    print("Hitrate avg: " + str(hits_avg / data))


if __name__ == '__main__':
    yf.pdr_override()
    dat = np.loadtxt("TestData/btcusd.csv")
    test_zhang(dat)
    # dat = [1.0, 0.5, 0.3, 0.7, 1.0, 0.5, 0.3, 0.8, 1.0, 0.4]


