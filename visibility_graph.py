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
import pmdarima as pm

def to_distance_weighted_graph(graph, time_series_data):
    for u, v in graph.edges:
        if v > u:
            graph[u][v]['weight'] = time_series_data[v] - time_series_data[u]
        else:
            graph[u][v]['weight'] = time_series_data[u] - time_series_data[v]

def neighbourhood_simmilarity_pr(matrix, neighbourhood, target_node):
    sum = 0
    if len(neighbourhood) == 0:
        return 0
    pageranks = nx.pagerank(matrix, alpha=0.6, tol=1e-02)
    node_pairs = [(target_node, x) for x in neighbourhood]
    for target, x in node_pairs:
        sum += 1 - abs(pageranks[target] - pageranks[x])
    return sum / len(neighbourhood)

def neighbourhood_simmilarity_sr(matrix, neighbourhood, target_node):
    sum = 0
    if len(neighbourhood) == 0:
        return 0
    node_pairs = [(target_node, x) for x in neighbourhood]
    for target, x in node_pairs:
        sum += nx.simrank_similarity(matrix, target, x, tolerance=0.1)
    return sum / len(neighbourhood)

def neighbourhood_simmilarity_aa(matrix, neighbourhood, target_node):
    sum = 0
    if len(neighbourhood) == 0:
        return 0
    node_pairs = [(target_node, x) for x in neighbourhood]
    preds = nx.adamic_adar_index(matrix, node_pairs)
    for u, v, p in preds:
        sum += p
    return sum / len(neighbourhood)

def neighbourhood_simmilarity_jc(matrix, neighbourhood, target_node):
    sum = 0
    if len(neighbourhood) == 0:
        return 0
    node_pairs = [(target_node, x) for x in neighbourhood]
    preds = nx.jaccard_coefficient(matrix, node_pairs)
    for u, v, p in preds:
        sum += p
    return sum / len(neighbourhood)


def generate_new_node_zhang_pagerank(time_series_values):
    # proposed method in this paper: https://www.sciencedirect.com/science/article/pii/S0378437117310622
    graph = NaturalVG()
    graph.build(time_series_values)
    matrix = graph.as_networkx()
    max_sim = 0
    max_sim_ix = 0
    last_node = len(matrix.nodes)-1
    pageranks = nx.pagerank(matrix, alpha=0.6, tol=1e-02)
    for node in matrix.nodes:
        if node == last_node:
            break
        temp_sim = 1 - abs(pageranks[node] - pageranks[last_node])

        if temp_sim > max_sim:
            max_sim = temp_sim
            max_sim_ix = node

    prediction = ((time_series_values[last_node] - time_series_values[max_sim_ix])/(last_node - max_sim_ix)
                  + time_series_values[last_node])

    return prediction

def generate_new_node_arima(time_series_values):
    arima_model = pm.ARIMA(order=(0, 1, 0), suppress_warnings=True)
    fit = arima_model.fit(time_series_values)
    arima_predict = fit.predict(n_periods=1)[0]
    return arima_predict

def generate_new_node_zhang_adamic_adar(time_series_values):
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
        preds = nx.adamic_adar_index(matrix, [(node, last_node)])
        for u, v, p in preds:
            temp_sim = p
            break
        if temp_sim > max_sim:
            max_sim = temp_sim
            max_sim_ix = node

    prediction = ((time_series_values[last_node] - time_series_values[max_sim_ix])/(last_node - max_sim_ix)
                  + time_series_values[last_node])

    return prediction

def generate_new_node_zhang_jaccard(time_series_values):
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
            break
        if temp_sim > max_sim:
            max_sim = temp_sim
            max_sim_ix = node

    prediction = ((time_series_values[last_node] - time_series_values[max_sim_ix])/(last_node - max_sim_ix)
                  + time_series_values[last_node])

    return prediction

def generate_new_node_zhang_simrank(time_series_values):
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
        temp_sim = nx.simrank_similarity(matrix, node, last_node, tolerance=0.1)

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
    # We add +1, to ensure continuation.
    num_of_edges = len(matrix.edges(last_node))
    if num_of_edges == 1:
        num_of_edges = 2
    indexes = []
    values = []
    values_ahead_difference = []
    for i in range(num_of_edges, 0, -1):
        for node in matrix.nodes:
            if len(matrix.edges(node)) == i and node != last_node:
                indexes.append(node)
                values.append(time_series_values[node])
                values_ahead_difference.append(time_series_values[node+1] - time_series_values[node])
            elif len(matrix.edges(node)) == num_of_edges:
                indexes.append(node)
                values.append(time_series_values[node])
        if len(values_ahead_difference) != 0:
            break

    indexes = np.array(indexes).reshape((-1, 1))
    values = np.array(values)
    if len(indexes) == 0 or len(values) == 0 or len(values_ahead_difference) == 0:
        return time_series_values[last_node]
    model = LinearRegression()
    model.fit(indexes, values)

    return (time_series_values[last_node] + (model.coef_ + (sum(values_ahead_difference)/len(values_ahead_difference))) / 2)[0]

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
    num_of_edges = len(matrix.edges(last_node))
    if num_of_edges == 1:
        num_of_edges = 2
    values_ahead_difference = []
    for i in range(num_of_edges, 0, -1):
        for node in matrix.nodes:
            if len(matrix.edges(node)) == i and node != last_node:
                values_ahead_difference.append(time_series_values[node+1] - time_series_values[node])
        if len(values_ahead_difference) != 0:
            break
    if len(values_ahead_difference) == 0:
        return time_series_values[last_node]


    prediction = time_series_values[last_node] + (sum(values_ahead_difference)/len(values_ahead_difference))

    return prediction

def generate_new_node_minimal_weighted_distance(time_series_values):
    graph = NaturalVG()
    graph.build(time_series_values)
    matrix = graph.as_networkx()
    to_distance_weighted_graph(matrix, time_series_values)

    last_node = len(matrix.nodes)-1
    last_node_edges = [(u, v) for u, v in list(matrix.edges(last_node)) if u > v]
    last_node_weights_sum = sum(matrix.get_edge_data(u, v)['weight']/len(last_node_edges) for u, v in last_node_edges)
    # Gledamo samo node za nazaj za primerjavo podobnosi.
    last_dist = np.inf
    closest_node = 0
    for node in matrix.nodes:
        if node == last_node:
            continue
        node_edges = [(u, v) for u, v in list(matrix.edges(node)) if u > v]
        node_weights_sum = sum(
            matrix.get_edge_data(u, v)['weight'] / len(node_edges) for u, v in node_edges)
        if node_weights_sum == 0:
            continue
        dist = abs(node_weights_sum - last_node_weights_sum)
        if last_dist > dist:
            closest_node = node
            last_dist = dist

    return time_series_values[last_node] + (time_series_values[closest_node+1] - time_series_values[closest_node])


def generate_new_node_max_clique(time_series_values):
    if len(time_series_values) <= 1:
        return time_series_values[0]
    graph = NaturalVG()
    graph.build(time_series_values)

    matrix = graph.as_networkx()

    last_node = len(matrix.nodes) - 1

    all_cliques = nx.find_cliques(matrix, [last_node])

    max_clique = []
    for clique in all_cliques:
        if len(clique) > len(max_clique):
            max_clique = clique
    values_ahead_difference = []
    for node in max_clique:
        if node != last_node:
            values_ahead_difference.append(time_series_values[node+1] - time_series_values[node])

    return time_series_values[last_node] + (sum(values_ahead_difference)/len(values_ahead_difference))


def generate_new_node_colouring(time_series_values):
    graph = NaturalVG()
    graph.build(time_series_values)

    matrix = graph.as_networkx()

    last_node = len(matrix.nodes) - 1

    colouring = nx.greedy_color(matrix)

    target_colour = colouring[last_node]
    same_color_nodes = (key for key, value in colouring.items() if value==target_colour)
    values_ahead_difference = []
    for node in same_color_nodes:
        if node != last_node:
            values_ahead_difference.append(time_series_values[node + 1] - time_series_values[node])

    if len(values_ahead_difference) == 0:
        return 2*time_series_values[last_node] - time_series_values[last_node-1]

    return time_series_values[last_node] + (sum(values_ahead_difference) / len(values_ahead_difference))


def generate_new_node_zhang_extended_jaccard(time_series_values, length=10):
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

def generate_new_node_zhang_extended_adamic_adar(time_series_values, length=10):
    # Extended method of the proposed method by taking into account multiple nodes.

    graph = NaturalVG()
    graph.build(time_series_values)

    matrix = graph.as_networkx()
    last_node = len(matrix.nodes)-1
    sims = [0 for x in range(last_node)]
    for node in matrix.nodes:
        if node == last_node:
            break
        preds = nx.adamic_adar_index(matrix, [(node, last_node)])
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

def generate_new_node_zhang_extended_pagerank(time_series_values, length=10):
    # Extended method of the proposed method by taking into account multiple nodes.

    graph = NaturalVG()
    graph.build(time_series_values)

    matrix = graph.as_networkx()
    last_node = len(matrix.nodes)-1
    sims = [0 for x in range(last_node)]
    pageranks = nx.pagerank(matrix, alpha=0.6, tol=1e-02)
    for node in matrix.nodes:
        if node == last_node:
            break
        temp_sim = 1 - abs(pageranks[node] - pageranks[last_node])
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

def generate_new_node_zhang_extended_simrank(time_series_values, length=10):
    # Extended method of the proposed method by taking into account multiple nodes.

    graph = NaturalVG()
    graph.build(time_series_values)

    matrix = graph.as_networkx()
    last_node = len(matrix.nodes)-1
    sims = [0 for x in range(last_node)]
    for node in matrix.nodes:
        if node == last_node:
            break
        temp_sim = nx.simrank_similarity(matrix, node, last_node, tolerance=0.1)
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

def generate_new_node_naive_sr(time_series_values, neighbourhood_size: int=20):
    graph = NaturalVG()
    graph.build(time_series_values)

    matrix = graph.as_networkx()
    new_node_index = len(matrix.nodes)
    max_neighbourhood_size = neighbourhood_size*3

    # Create new node.
    neighbourhood = list(range(new_node_index))[-neighbourhood_size:]
    matrix.add_node(new_node_index)
    matrix.add_edge(new_node_index - 1, new_node_index)

    # Add nodes.
    for node in neighbourhood:
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
            sim = neighbourhood_simmilarity_sr(matrix, neighbourhood, new_node_index)
            if sim <= previous_sim:
                matrix.remove_edge(new_node_index, edge[1])
                sim = neighbourhood_simmilarity_sr(matrix, neighbourhood, new_node_index)
            previous_sim = sim

    # Remove nodes.
    previous_sim = neighbourhood_simmilarity_sr(matrix, neighbourhood, new_node_index)
    new_edges = list(matrix.edges(new_node_index))
    for edge in new_edges:
        matrix.remove_edge(new_node_index, edge[1])
        sim = neighbourhood_simmilarity_sr(matrix, neighbourhood, new_node_index)
        if sim <= previous_sim:
            matrix.add_edge(new_node_index, edge[1])
            sim = neighbourhood_simmilarity_sr(matrix, neighbourhood, new_node_index)
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
        if edge[1] >= len(time_series_values)-1:
            continue
        predicted_value += (time_series_values[edge[1]+1] - time_series_values[edge[1]]) / len(new_edges)

    time_series_values = np.append(time_series_values, time_series_values[-1] + predicted_value)

    return time_series_values[-1]

def generate_new_node_naive_aa(time_series_values, neighbourhood_size: int=10):
    graph = NaturalVG()
    graph.build(time_series_values)

    matrix = graph.as_networkx()
    new_node_index = len(matrix.nodes)
    max_neighbourhood_size = neighbourhood_size*3

    # Create new node.
    neighbourhood = list(range(new_node_index))[-neighbourhood_size:]
    matrix.add_node(new_node_index)
    matrix.add_edge(new_node_index - 1, new_node_index)

    # Add nodes.
    for node in neighbourhood:
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
            sim = neighbourhood_simmilarity_aa(matrix, neighbourhood, new_node_index)
            if sim <= previous_sim:
                matrix.remove_edge(new_node_index, edge[1])
                sim = neighbourhood_simmilarity_aa(matrix, neighbourhood, new_node_index)
            previous_sim = sim

    # Remove nodes.
    previous_sim = neighbourhood_simmilarity_aa(matrix, neighbourhood, new_node_index)
    new_edges = list(matrix.edges(new_node_index))
    for edge in new_edges:
        matrix.remove_edge(new_node_index, edge[1])
        sim = neighbourhood_simmilarity_aa(matrix, neighbourhood, new_node_index)
        if sim <= previous_sim:
            matrix.add_edge(new_node_index, edge[1])
            sim = neighbourhood_simmilarity_aa(matrix, neighbourhood, new_node_index)
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
        if edge[1] >= len(time_series_values)-1:
            continue
        predicted_value += (time_series_values[edge[1]+1] - time_series_values[edge[1]]) / len(new_edges)

    time_series_values = np.append(time_series_values, time_series_values[-1] + predicted_value)

    return time_series_values[-1]

def generate_new_node_naive_pr(time_series_values, neighbourhood_size: int=10):
    graph = NaturalVG()
    graph.build(time_series_values)

    matrix = graph.as_networkx()
    new_node_index = len(matrix.nodes)
    max_neighbourhood_size = neighbourhood_size*3

    # Create new node.
    neighbourhood = list(range(new_node_index))[-neighbourhood_size:]
    matrix.add_node(new_node_index)
    matrix.add_edge(new_node_index - 1, new_node_index)

    # Add nodes.
    for node in neighbourhood:
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
            sim = neighbourhood_simmilarity_pr(matrix, neighbourhood, new_node_index)
            if sim <= previous_sim:
                matrix.remove_edge(new_node_index, edge[1])
                sim = neighbourhood_simmilarity_pr(matrix, neighbourhood, new_node_index)
            previous_sim = sim

    # Remove nodes.
    previous_sim = neighbourhood_simmilarity_pr(matrix, neighbourhood, new_node_index)
    new_edges = list(matrix.edges(new_node_index))
    for edge in new_edges:
        matrix.remove_edge(new_node_index, edge[1])
        sim = neighbourhood_simmilarity_pr(matrix, neighbourhood, new_node_index)
        if sim <= previous_sim:
            matrix.add_edge(new_node_index, edge[1])
            sim = neighbourhood_simmilarity_pr(matrix, neighbourhood, new_node_index)
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
        if edge[1] >= len(time_series_values)-1:
            continue
        predicted_value += (time_series_values[edge[1]+1] - time_series_values[edge[1]]) / len(new_edges)

    time_series_values = np.append(time_series_values, time_series_values[-1] + predicted_value)

    return time_series_values[-1]

def generate_new_node_naive_jc(time_series_values, neighbourhood_size: int=10):
    graph = NaturalVG()
    graph.build(time_series_values)

    matrix = graph.as_networkx()
    new_node_index = len(matrix.nodes)
    max_neighbourhood_size = neighbourhood_size*3

    # Create new node.
    neighbourhood = list(range(new_node_index))[-neighbourhood_size:]
    matrix.add_node(new_node_index)
    matrix.add_edge(new_node_index - 1, new_node_index)

    # Add nodes.
    for node in neighbourhood:
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
            sim = neighbourhood_simmilarity_jc(matrix, neighbourhood, new_node_index)
            if sim <= previous_sim:
                matrix.remove_edge(new_node_index, edge[1])
                sim = neighbourhood_simmilarity_jc(matrix, neighbourhood, new_node_index)
            previous_sim = sim

    # Remove nodes.
    previous_sim = neighbourhood_simmilarity_jc(matrix, neighbourhood, new_node_index)
    new_edges = list(matrix.edges(new_node_index))
    for edge in new_edges:
        matrix.remove_edge(new_node_index, edge[1])
        sim = neighbourhood_simmilarity_jc(matrix, neighbourhood, new_node_index)
        if sim <= previous_sim:
            matrix.add_edge(new_node_index, edge[1])
            sim = neighbourhood_simmilarity_jc(matrix, neighbourhood, new_node_index)
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
        if edge[1] >= len(time_series_values)-1:
            continue
        predicted_value += (time_series_values[edge[1]+1] - time_series_values[edge[1]]) / len(new_edges)

    time_series_values = np.append(time_series_values, time_series_values[-1] + predicted_value)

    return time_series_values[-1]
