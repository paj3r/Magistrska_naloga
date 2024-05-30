import datetime as dt
import os

import networkx as nx
import numpy as np
import pandas_datareader.data as pdr
import pandas as pd
import yfinance as yf
from ts2vg import NaturalVG, HorizontalVG
import matplotlib.pyplot as plt
from heapq import nlargest
import matplotlib.patches as patches
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse, mean_absolute_percentage_error as mape
from visibility_graph import *
from ts_to_vg import plot_ts_visibility  # Plotting function for visibility graph


def print_statistics(actual, positive, negative, average, output, test_name):
    countnan = 0
    for value in positive:
        if np.isnan(value):
            countnan += 1
        else:
            break
    window_size = 10
    positive_plot = positive[countnan:] - actual[countnan:]
    negative_plot = negative[countnan:] - actual[countnan:]
    average_plot = average[countnan:] - actual[countnan:]
    positive_plot_cumsum = np.cumsum(positive_plot)
    negative_plot_cumsum = np.cumsum(negative_plot)
    average_plot_cumsum = np.cumsum(average_plot)
    pos_plot_averages = []
    neg_plot_averages = []
    pos_plot_max = []
    neg_plot_max = []
    pos_plot_min = []
    neg_plot_min = []
    for i in range(0, len(positive_plot), window_size):
        max = -np.inf
        min = np.inf
        avg = 0
        for j in range(0, window_size):
            if i+j >= len(positive_plot):
                break
            temp_value = positive_plot[i+j]
            if temp_value > max:
                max = temp_value
            if temp_value < min:
                min = temp_value
            avg += temp_value
        avg = avg / window_size
        pos_plot_averages.append(avg)
        pos_plot_max.append(max)
        pos_plot_min.append(min)
    for i in range(0, len(negative_plot), window_size):
        max = -np.inf
        min = np.inf
        avg = 0
        for j in range(0, window_size):
            if i+j >= len(negative_plot):
                break
            temp_value = negative_plot[i+j]
            if temp_value > max:
                max = temp_value
            if temp_value < min:
                min = temp_value
            avg += temp_value
        avg = avg / window_size
        neg_plot_averages.append(avg)
        neg_plot_max.append(max)
        neg_plot_min.append(min)
    pos_minmax = [pos_plot_min, pos_plot_max]
    neg_minmax = [neg_plot_min, neg_plot_max]
    plt.plot(pos_plot_averages, label="Positive predictions", alpha=0.8, color="green", linewidth=0.5)
    plt.plot(pos_plot_max, label="Positive predictions max", alpha=0.8, color="limegreen", linewidth=0.5)
    plt.plot(pos_plot_min, label="Positive predictions min", alpha=0.8, color="limegreen", linewidth=0.5)
    # plt.plot()
    # plt.legend()
    # plt.title(test_name)
    # plt.savefig(f"./Results/Pictures/{test_name}_positive.svg", format='svg', dpi=300)
    plt.plot(neg_plot_averages, label="Negative predictions", alpha=0.8, color="red", linewidth=0.5)
    plt.plot(neg_plot_max, label="Negative predictions max", alpha=0.8, color="lightcoral", linewidth=0.5)
    plt.plot(neg_plot_min, label="Negative predictions min", alpha=0.8, color="lightcoral", linewidth=0.5)
    #plt.plot(average_plot, label="Average predictions", alpha=0.5, linewidth=0.25)
    plt.plot()
    plt.legend()
    plt.title(test_name)
    plt.savefig(f"./Results/Pictures/{test_name}.svg", format='svg', dpi=300)
    plt.show()
    plt.plot(positive_plot_cumsum, label="Positive cumulative predictions", alpha=0.7)
    plt.plot(negative_plot_cumsum, label="Negative cumulative predictions", alpha=0.7)
    plt.plot(average_plot_cumsum, label="Average cumulative predictions", alpha=0.7)
    plt.plot()
    plt.legend()
    plt.title(test_name)
    plt.savefig(f"./Results/Pictures/{test_name}_cumulative.svg", format='svg', dpi=300)
    plt.show()
    plt.plot(positive[countnan:], label="Positive predictions", alpha=0.7, linewidth=0.5)
    plt.plot(negative[countnan:], label="Negative predictions", alpha=0.7, linewidth=0.5)
    plt.plot(average[countnan:], label="Average predictions", alpha=0.7, linewidth=0.5)
    plt.plot(actual[countnan:], label="Actual values", alpha=0.7, linewidth=0.5)
    plt.plot()
    plt.legend()
    plt.title(test_name)
    plt.savefig(f"./Results/Pictures/{test_name}_predictions.svg", format='svg', dpi=300)
    plt.show()
    # remove nan.
    positive = positive[~np.isnan(positive)]
    negative = negative[~np.isnan(negative)]
    average = average[~np.isnan(average)]
    output += "\n"
    output += f"Positive MAE: {mae(actual[:len(positive)], positive)}\n"
    output += f"Negative MAE: {mae(actual[:len(negative)], negative)}\n"
    output += f"Average MAE: {mae(actual[:len(average)], average)}\n\n"
    output += f"Positive MSE: {mse(actual[:len(positive)], positive)}\n"
    output += f"Negative MSE: {mse(actual[:len(negative)], negative)}\n"
    output += f"Average MSE: {mse(actual[:len(average)], average)}\n\n"
    output += f"Positive MAPE: {mape(actual[:len(positive)], positive)}\n"
    output += f"Negative MAPE: {mape(actual[:len(negative)], negative)}\n"
    output += f"Average MAPE: {mape(actual[:len(average)], average)}\n"

    return output

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

def test_minimal_weighted_distance(dat):
    neg_dat = [-x for x in dat]
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
        rolling_dat_neg = [-x for x in rolling_dat]
        new_value = generate_node_minimal_weighted_distance(rolling_dat)
        new_value_neg = -generate_node_minimal_weighted_distance(rolling_dat_neg)
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

def test_generator_function_optimisation(generator_function, dat, filename_prefix):
    output = ""
    length_range = range(2, 30)
    start_offset = 30
    neg_dat = [-x for x in dat]

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
        output += f"Length size: {length} \n"
        predictions = np.empty(start_offset)
        neg_predictions = np.empty(start_offset)
        predictions[:] = np.nan
        neg_predictions[:] = np.nan
        for i in range(start_offset, len(dat)):
            rolling_dat = dat[:i]
            rolling_neg_dat = neg_dat[:i]
            new_value = generator_function(rolling_dat, length)
            new_value_neg = -generator_function(rolling_neg_dat, length)
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

        output += f"Data length: {data}\n"
        output += f"Hits: {hits_pos}\n"
        output += f"Hitrate pos: {str(hits_pos / data)}\n"
        output += f"Hits neg: {hits_neg}\n"
        output += f"Hitrate neg: {str(hits_neg / data)}\n"
        output += f"Hits avg: {hits_avg}\n"
        output += f"Hitrate avg: {str(hits_avg / data)}\n\n"

    output += f"Max positive: {max_pos} At neighbourhood size: {max_pos_ix}\n"
    output += f"Max negative: {max_neg} At neighbourhood size: {max_neg_ix}\n"
    output += f"Max average: {max_avg} At neighbourhood size: {max_avg_ix}\n"
    output = print_statistics(dat, max_pos_preds, max_neg_preds, max_avg_preds, output, filename_prefix)

    f = open(f"./Results/{filename_prefix}.txt", "w")
    f.write(output)
    f.close()

def test_generator_function(generator_function, dat, filename_prefix):
    output = ""
    neg_dat = [-x for x in dat]
    start_offset = 30

    predictions = np.empty(start_offset)
    predictions[:] = np.nan
    neg_predictions = np.empty(start_offset)
    neg_predictions[:] = np.nan
    # 30 days of learning period.
    for i in range(start_offset, len(dat)):
        rolling_dat = dat[:i]
        rolling_dat_neg = neg_dat[:i]
        new_value = generator_function(rolling_dat)
        new_value_neg = -generator_function(rolling_dat_neg)
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

    output += f"Data length: {data}\n"
    output += f"Hits: {hits_pos}\n"
    output += f"Hitrate pos: {str(hits_pos / data)}\n"
    output += f"Hits neg: {hits_neg}\n"
    output += f"Hitrate neg: {str(hits_neg / data)}\n"
    output += f"Hits avg: {hits_avg}\n"
    output += f"Hitrate avg: {str(hits_avg / data)}\n"
    avg_predictions = (predictions + neg_predictions) / 2
    output = print_statistics(dat, predictions, neg_predictions, avg_predictions, output, filename_prefix)

    if not os.path.exists(f"./Results/{filename_prefix}"):
        os.makedirs(f"./Results/{filename_prefix}")

    f = open(f"./Results/{filename_prefix}/statistics.txt", "w")
    f.write(output)
    f.close()

    output_data = {"Actual" : dat[start_offset:], "Positive" : predictions[start_offset:],
                   "Negative" : neg_predictions[start_offset:], "Average": avg_predictions[start_offset:]}
    df = pd.DataFrame(output_data)
    df.to_excel(f"./Results/{filename_prefix}/data.xlsx", index=False)

if __name__ == '__main__':
    yf.pdr_override()
    for file in os.listdir("TestData"):
        data = np.loadtxt(f"TestData/{file}")
        test_generator_function_optimisation(generate_new_node_zhang_extended, data,f"zhang_extended_{file.split('.')[0]}")
        #test_generator_function(generate_new_node_zhang, data, f"zhang_{file.split('.')[0]}")
        # dat = [1.0, 0.5, 0.3, 0.7, 1.0, 0.5, 0.3, 0.8, 1.0, 0.4]


