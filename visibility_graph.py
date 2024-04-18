import datetime as dt

import networkx as nx
import numpy as np
import pandas_datareader.data as pdr
import yfinance as yf
from ts2vg import NaturalVG, HorizontalVG
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from heapq import nlargest

def neighbourhood_simmilarity(matrix, neighbourhood, target_node):
    sum = 0
    if len(neighbourhood) == 0:
        return 0
    node_pairs = [(target_node, x) for x in neighbourhood.nodes]
    preds = nx.jaccard_coefficient(matrix, node_pairs)
    for u, v, p in preds:
        sum += p
    return sum / len(neighbourhood)


def generate_new_node_zhang(time_series_values, ahead=1):
    # proposed method in this paper: https://www.sciencedirect.com/science/article/pii/S0378437117310622
    graph = NaturalVG()
    graph.build(time_series_values)

    matrix = graph.as_networkx()

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

def generate_new_node_isomorphism_trend_linear(time_series_values, ahead=1):
    graph = NaturalVG()
    graph.build(time_series_values)

    matrix = graph.as_networkx()

    last_node = len(matrix.nodes)-1
    num_of_edges = len(matrix.edges(last_node))
    indexes = []
    values = []
    for node in matrix.nodes:
        if len(matrix.edges(node)) == num_of_edges:
            indexes.append(node)
            values.append(time_series_values[node])
    indexes = np.array(indexes).reshape((-1, 1))
    values = np.array(values)
    model = LinearRegression()
    model.fit(indexes, values)
    prediction = model.predict(np.array([last_node+ahead]).reshape(-1, 1))

    return prediction[0]

def generate_new_node_isomorphism(time_series_values, positive=True):
    if not positive:
        data = [-x for x in time_series_values]
    else:
        data = time_series_values
    graph = NaturalVG()
    graph.build(data)

    matrix = graph.as_networkx()

    last_node = len(matrix.nodes)-1
    # We add +1, to ensure continuation.
    num_of_edges = len(matrix.edges(last_node)) + 1
    values_ahead_difference = []
    for i in range(num_of_edges, 0, -1):
        for node in matrix.nodes:
            if len(matrix.edges(node)) == i and node != last_node:
                values_ahead_difference.append(time_series_values[node+1] - time_series_values[node])
        if len(values_ahead_difference) != 0:
            break


    prediction = time_series_values[last_node] + (sum(values_ahead_difference)/len(values_ahead_difference))

    return prediction

def generate_new_node_zhang_extended(time_series_values, length):
    # Extended method of the proposed method by taking into account multiple nodes.

    graph = NaturalVG()
    graph.build(time_series_values)

    matrix = graph.as_networkx()
    last_node = len(matrix.nodes)-1
    sims = [0 for x in range(last_node)]
    for node in matrix.nodes:
        if node == last_node:
            break
        preds = nx.jaccard_coefficient(matrix, [(node, last_node)])
        for u, v, p in preds:
            temp_sim = p
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

def generate_new_node_naive(time_series_values, neighbourhood_size: int):
    graph = NaturalVG()
    graph.build(time_series_values)

    matrix = graph.as_networkx()
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

    return time_series_values[-1]

