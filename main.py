import datetime as dt

import networkx as nx
import numpy as np
import pandas_datareader.data as pdr
import yfinance as yf
from ts2vg import NaturalVG, HorizontalVG
import matplotlib.pyplot as plt
from heapq import nlargest
import matplotlib.patches as patches
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse, mean_absolute_percentage_error as mape
from visibility_graph import generate_new_node_colouring, generate_new_node_isomorphism_trend_linear, generate_new_node_isomorphism, to_distance_weighted_graph, generate_new_node_max_clique

from ts_to_vg import plot_ts_visibility  # Plotting function for visibility graph


def print_statistics(actual, positive, negative, average):
    plt.plot(actual, label="Original data")
    plt.plot(positive, label="Positive predictions")
    plt.plot(negative, label="Negative predictions")
    plt.plot(average, label="Average predictions")
    plt.legend()
    plt.show()
    # remove nan.
    positive = positive[~np.isnan(positive)]
    negative = negative[~np.isnan(negative)]
    average = average[~np.isnan(average)]
    print("\n")
    print("Positive MAE: ", mae(actual[:len(positive)], positive))
    print("Negative MAE: ", mae(actual[:len(negative)], negative))
    print("Average MAE: ", mae(actual[:len(average)], average))
    print("\n")
    print("Positive MSE:", mse(actual[:len(positive)], positive))
    print("Negative MSE:", mse(actual[:len(negative)], negative))
    print("Average MSE:", mse(actual[:len(average)], average))
    print("\n")
    print("Positive MAPE:", mape(actual[:len(positive)], positive))
    print("Negative MAPE:", mape(actual[:len(negative)], negative))
    print("Average MAPE:", mape(actual[:len(average)], average))

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
    #preds = nx.adamic_adar_index(matrix, node_pairs)
    #for u, v, p in preds:
    #    sum += p
    preds = nx.google_matrix(matrix)
    for pair in node_pairs:
        sum += preds[pair[0]][pair[1]]
    return sum / len(neighbourhood)


def generate_new_node_zhang(time_series_values, matrix:nx.classes.graph.Graph, ahead=1):
    # proposed method in this paper: https://www.sciencedirect.com/science/article/pii/S0378437117310622
    max_sim = 0
    max_sim_ix = 0
    last_node = len(matrix.nodes)-1
    for node in matrix.nodes:
        if node == last_node:
            break
        #preds = nx.adamic_adar_index(matrix, [(node, last_node)])
        preds = nx.google_matrix(matrix)
        temp_sim = preds[last_node][node]
        if temp_sim > max_sim:
            max_sim = temp_sim
            max_sim_ix = node

    prediction = ((time_series_values[last_node] - time_series_values[max_sim_ix])/(last_node - max_sim_ix)
                  + time_series_values[last_node])

    return prediction

def generate_new_node_zhang_extended(time_series_values, matrix:nx.classes.graph.Graph, length):
    # Extended method of the proposed method by taking into account multiple nodes.
    last_node = len(matrix.nodes)-1
    sims = [0 for x in range(last_node)]
    for node in matrix.nodes:
        if node == last_node:
            break
        preds = nx.google_matrix(matrix)
        temp_sim = preds[last_node][node]
        sims[node] = temp_sim
    n_sims = nlargest(length, sims)
    n_sims_ix = nlargest(length, range(len(sims)), sims.__getitem__)

    sum_sims = sum(n_sims)
    if sum_sims == 0:
        return 0
    prediction = 0
    for sim_ix in n_sims_ix:
        prediction += ((time_series_values[last_node] - time_series_values[sim_ix])/(last_node - sim_ix)
                      + time_series_values[last_node])*(sims[sim_ix]/sum_sims)
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
    start_offset = 30
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
    max_pos_preds = []
    max_neg = 0
    max_neg_ix = 0
    max_neg_preds = []
    max_avg = 0
    max_avg_ix = 0
    max_avg_preds = []
    for neighbourhood_size in neighbourhood_range:
        print("Neighbourhood size:", neighbourhood_size)
        predictions = np.empty(start_offset)
        predictions[:] = np.nan
        neg_predictions = np.empty(start_offset)
        neg_predictions[:] = np.nan
        for i in range(start_offset, len(dat)):
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
        for i in range(start_offset, len(dat)):
            if dat[i - 1] < predictions[i] and dat[i - 1] < dat[i]:
                hits_pos += 1
            elif dat[i - 1] > predictions[i] and dat[i - 1] > dat[i]:
                hits_pos += 1
            data += 1
        if max_pos < hits_pos / data:
            max_pos = hits_pos / data
            max_pos_ix = neighbourhood_size
            max_pos_preds = predictions
        hits_neg = 0
        # Check negative hitrate
        for i in range(start_offset, len(dat)):
            if dat[i - 1] < neg_predictions[i] and dat[i - 1] < dat[i]:
                hits_neg += 1
            elif dat[i - 1] > neg_predictions[i] and dat[i - 1] > dat[i]:
                hits_neg += 1
        if max_neg < hits_neg / data:
            max_neg = hits_neg / data
            max_neg_ix = neighbourhood_size
            max_neg_preds = neg_predictions

        hits_avg = 0
        # Check negative hitrate
        for i in range(start_offset, len(dat)):
            val = (neg_predictions[i] + predictions[i]) / 2
            if dat[i - 1] < val and dat[i - 1] < dat[i]:
                hits_avg += 1
            elif dat[i - 1] > val and dat[i - 1] > dat[i]:
                hits_avg += 1
        if max_avg < hits_avg / data:
            max_avg = hits_avg / data
            max_avg_ix = neighbourhood_size
            max_avg_preds = (max_pos_preds + max_neg_preds) / 2
        print("Data length: ", data)
        print("Hits: ", hits_pos)
        print("Hitrate pos: " + str(hits_pos / data))
        print("Hits neg: ", hits_neg)
        print("Hitrate neg: " + str(hits_neg / data))
        print("Hits avg: ", hits_avg)
        print("Hitrate avg: " + str(hits_avg / data))
        print("\n")
        pass
    print("Max positive: ", max_pos, " At neighbourhood size:", max_pos_ix)
    print("Max negative: ", max_neg, " At neighbourhood size:", max_neg_ix)
    print("Max average: ", max_avg, " At neighbourhood size:", max_avg_ix)
    print_statistics(dat, max_pos_preds, max_neg_preds, max_avg_preds)


def test_isomorphism_trend(dat):
    start_offset = 30
    neg_dat = [-x for x in dat]

    hits = 0
    data = 0
    preds = []
    predictions = np.empty(start_offset)
    predictions[:] = np.nan
    neg_predictions = np.empty(start_offset)
    neg_predictions[:] = np.nan
    # 30 days of learning period.
    for i in range(start_offset, len(dat)):
        rolling_dat = dat[:i]
        rolling_neg = neg_dat[:i]
        new_value = generate_new_node_isomorphism_trend_linear(rolling_dat)
        new_value_neg = -generate_new_node_isomorphism_trend_linear(rolling_neg)
        predictions = np.append(predictions, new_value)
        neg_predictions = np.append(neg_predictions, new_value_neg)

    # Check positive hitrate
    hits_pos = 0
    data = 0
    for i in range(start_offset, len(dat)):
        if dat[i - 1] < predictions[i] and dat[i - 1] < dat[i]:
            hits_pos += 1
        elif dat[i - 1] > predictions[i] and dat[i - 1] > dat[i]:
            hits_pos += 1
        data += 1

    hits_neg = 0
    # Check negative hitrate
    for i in range(start_offset, len(dat)):
        if dat[i - 1] < neg_predictions[i] and dat[i - 1] < dat[i]:
            hits_neg += 1
        elif dat[i - 1] > neg_predictions[i] and dat[i - 1] > dat[i]:
            hits_neg += 1

    hits_avg = 0
    # Check negative hitrate
    for i in range(start_offset, len(dat)):
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
    avg_predictions = (predictions + neg_predictions) / 2
    print_statistics(dat, predictions, neg_predictions, avg_predictions)

def test_isomorphism(dat):
    start_offset = 30

    hits = 0
    data = 0
    preds = []
    predictions = np.empty(start_offset)
    predictions[:] = np.nan
    neg_predictions = np.empty(start_offset)
    neg_predictions[:] = np.nan
    # 30 days of learning period.
    for i in range(start_offset, len(dat)):
        rolling_dat = dat[:i]
        new_value = generate_new_node_isomorphism(rolling_dat)
        new_value_neg = generate_new_node_isomorphism(rolling_dat, False)
        predictions = np.append(predictions, new_value)
        neg_predictions = np.append(neg_predictions, new_value_neg)

    # Check positive hitrate
    hits_pos = 0
    data = 0
    for i in range(start_offset, len(dat)):
        if dat[i - 1] < predictions[i] and dat[i - 1] < dat[i]:
            hits_pos += 1
        elif dat[i - 1] > predictions[i] and dat[i - 1] > dat[i]:
            hits_pos += 1
        data += 1

    hits_neg = 0
    # Check negative hitrate
    for i in range(start_offset, len(dat)):
        if dat[i - 1] < neg_predictions[i] and dat[i - 1] < dat[i]:
            hits_neg += 1
        elif dat[i - 1] > neg_predictions[i] and dat[i - 1] > dat[i]:
            hits_neg += 1

    hits_avg = 0
    # Check negative hitrate
    for i in range(start_offset, len(dat)):
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
    avg_predictions = (predictions + neg_predictions) / 2
    print_statistics(dat, predictions, neg_predictions, avg_predictions)

def test_zhang(dat):
    neg_dat = [-x for x in dat]
    start_offset = 30
    graph = NaturalVG()
    graph.build(dat)
    neg_graph = NaturalVG()
    neg_graph.build(neg_dat)
    network = graph.as_networkx()
    neg_network = neg_graph.as_networkx()
    to_distance_weighted_graph(network, dat)
    to_distance_weighted_graph(neg_network, neg_dat)

    hits = 0
    data = 0
    preds = []
    predictions = np.empty(start_offset)
    predictions[:] = np.nan
    neg_predictions = np.empty(start_offset)
    neg_predictions[:] = np.nan
    # 30 days of learning period.
    for i in range(start_offset, len(dat)):
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
    for i in range(start_offset, len(dat)):
        if dat[i - 1] < predictions[i] and dat[i - 1] < dat[i]:
            hits_pos += 1
        elif dat[i - 1] > predictions[i] and dat[i - 1] > dat[i]:
            hits_pos += 1
        data += 1

    hits_neg = 0
    # Check negative hitrate
    for i in range(start_offset, len(dat)):
        if dat[i - 1] < neg_predictions[i] and dat[i - 1] < dat[i]:
            hits_neg += 1
        elif dat[i - 1] > neg_predictions[i] and dat[i - 1] > dat[i]:
            hits_neg += 1

    hits_avg = 0
    # Check negative hitrate
    for i in range(start_offset, len(dat)):
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
    avg_predictions = (predictions + neg_predictions) / 2
    print_statistics(dat, predictions, neg_predictions, avg_predictions)


def test_max_clique(dat):
    neg_dat = [-x for x in dat]
    start_offset = 30
    graph = NaturalVG()
    graph.build(dat)
    neg_graph = NaturalVG()
    neg_graph.build(neg_dat)
    network = graph.as_networkx()
    neg_network = neg_graph.as_networkx()
    to_distance_weighted_graph(network, dat)
    to_distance_weighted_graph(neg_network, neg_dat)

    hits = 0
    data = 0
    preds = []
    predictions = np.empty(start_offset)
    predictions[:] = np.nan
    neg_predictions = np.empty(start_offset)
    neg_predictions[:] = np.nan
    # 30 days of learning period.
    for i in range(start_offset, len(dat)):
        rolling_dat = dat[:i]
        rolling_dat_neg = [-x for x in rolling_dat]
        new_value = generate_new_node_max_clique(rolling_dat)
        new_value_neg = -generate_new_node_max_clique(rolling_dat_neg)
        predictions = np.append(predictions, new_value)
        neg_predictions = np.append(neg_predictions, new_value_neg)

    # Check positive hitrate
    hits_pos = 0
    data = 0
    for i in range(start_offset, len(dat)):
        if dat[i - 1] < predictions[i] and dat[i - 1] < dat[i]:
            hits_pos += 1
        elif dat[i - 1] > predictions[i] and dat[i - 1] > dat[i]:
            hits_pos += 1
        data += 1

    hits_neg = 0
    # Check negative hitrate
    for i in range(start_offset, len(dat)):
        if dat[i - 1] < neg_predictions[i] and dat[i - 1] < dat[i]:
            hits_neg += 1
        elif dat[i - 1] > neg_predictions[i] and dat[i - 1] > dat[i]:
            hits_neg += 1

    hits_avg = 0
    # Check negative hitrate
    for i in range(start_offset, len(dat)):
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
    avg_predictions = (predictions + neg_predictions) / 2
    print_statistics(dat, predictions, neg_predictions, avg_predictions)

def test_colouring(dat):
    neg_dat = [-x for x in dat]
    start_offset = 30
    graph = NaturalVG()
    graph.build(dat)
    neg_graph = NaturalVG()
    neg_graph.build(neg_dat)
    network = graph.as_networkx()
    neg_network = neg_graph.as_networkx()
    to_distance_weighted_graph(network, dat)
    to_distance_weighted_graph(neg_network, neg_dat)

    hits = 0
    data = 0
    preds = []
    predictions = np.empty(start_offset)
    predictions[:] = np.nan
    neg_predictions = np.empty(start_offset)
    neg_predictions[:] = np.nan
    # 30 days of learning period.
    for i in range(start_offset, len(dat)):
        rolling_dat = dat[:i]
        rolling_dat_neg = [-x for x in rolling_dat]
        new_value = generate_new_node_colouring(rolling_dat)
        new_value_neg = -generate_new_node_colouring(rolling_dat_neg)
        predictions = np.append(predictions, new_value)
        neg_predictions = np.append(neg_predictions, new_value_neg)

    # Check positive hitrate
    hits_pos = 0
    data = 0
    for i in range(start_offset, len(dat)):
        if dat[i - 1] < predictions[i] and dat[i - 1] < dat[i]:
            hits_pos += 1
        elif dat[i - 1] > predictions[i] and dat[i - 1] > dat[i]:
            hits_pos += 1
        data += 1

    hits_neg = 0
    # Check negative hitrate
    for i in range(start_offset, len(dat)):
        if dat[i - 1] < neg_predictions[i] and dat[i - 1] < dat[i]:
            hits_neg += 1
        elif dat[i - 1] > neg_predictions[i] and dat[i - 1] > dat[i]:
            hits_neg += 1

    hits_avg = 0
    # Check negative hitrate
    for i in range(start_offset, len(dat)):
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
    avg_predictions = (predictions + neg_predictions) / 2
    print_statistics(dat, predictions, neg_predictions, avg_predictions)



def test_zhang_extended(dat):
    length_range = range(1, 30)
    start_offset = 30
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
    max_pos_preds = []
    max_neg = 0
    max_neg_ix = 0
    max_neg_preds = []
    max_avg = 0
    max_avg_ix = 0
    max_avg_preds = []
    for length in length_range:
        print("Length size:", length)
        predictions = np.zeros(start_offset)
        neg_predictions = np.zeros(start_offset)
        for i in range(start_offset, len(dat)):
            rolling_dat = dat[:i]
            rolling_network = network.subgraph(list(range(0, i))).copy()
            rolling_neg_network = neg_network.subgraph(list(range(0, i))).copy()
            new_value = generate_new_node_zhang_extended(rolling_dat, rolling_network, length)
            new_value_neg = generate_new_node_zhang_extended(rolling_dat, rolling_neg_network, length)
            predictions = np.append(predictions, new_value)
            neg_predictions = np.append(neg_predictions, new_value_neg)

        # Check positive hitrate
        hits_pos = 0
        data = 0
        for i in range(start_offset, len(dat)):
            if dat[i - 1] < predictions[i] and dat[i - 1] < dat[i]:
                hits_pos += 1
            elif dat[i - 1] > predictions[i] and dat[i - 1] > dat[i]:
                hits_pos += 1
            data += 1
        if max_pos < hits_pos / data:
            max_pos = hits_pos / data
            max_pos_ix = length
            max_pos_preds = predictions

        hits_neg = 0
        # Check negative hitrate
        for i in range(start_offset, len(dat)):
            if dat[i - 1] < neg_predictions[i] and dat[i - 1] < dat[i]:
                hits_neg += 1
            elif dat[i - 1] > neg_predictions[i] and dat[i - 1] > dat[i]:
                hits_neg += 1
        if max_neg < hits_neg / data:
            max_neg = hits_neg / data
            max_neg_ix = length
            max_neg_preds = neg_predictions

        hits_avg = 0
        # Check negative hitrate
        for i in range(start_offset, len(dat)):
            val = (neg_predictions[i] + predictions[i]) / 2
            if dat[i - 1] < val and dat[i - 1] < dat[i]:
                hits_avg += 1
            elif dat[i - 1] > val and dat[i - 1] > dat[i]:
                hits_avg += 1
        if max_avg < hits_avg / data:
            max_avg = hits_avg / data
            max_avg_ix = length
            max_avg_preds = (max_pos_preds + max_neg_preds) / 2

        print("Data length: ", data)
        print("Hits: ", hits_pos)
        print("Hitrate pos: " + str(hits_pos / data))
        print("Hits neg: ", hits_neg)
        print("Hitrate neg: " + str(hits_neg / data))
        print("Hits avg: ", hits_avg)
        print("Hitrate avg: " + str(hits_avg / data))
        print("\n")
        pass
    print("Max positive: ", max_pos, " At neighbourhood size:", max_pos_ix)
    print("Max negative: ", max_neg, " At neighbourhood size:", max_neg_ix)
    print("Max average: ", max_avg, " At neighbourhood size:", max_avg_ix)
    print_statistics(dat, max_pos_preds, max_neg_preds, max_avg_preds)

if __name__ == '__main__':
    yf.pdr_override()
    data = np.loadtxt("TestData/eurusd.csv")
    test_zhang_extended(data)
    # dat = [1.0, 0.5, 0.3, 0.7, 1.0, 0.5, 0.3, 0.8, 1.0, 0.4]


