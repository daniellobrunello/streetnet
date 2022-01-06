# coding: utf-8
# https://brooksandrew.github.io/simpleblog/articles/intro-to-graph-optimization-solving-cpp/#intro-to-graph-optimization-with-networkx-in-python

import itertools
import math
import sys
import json
from datetime import timedelta, datetime

from shapely.geometry import Polygon
from tqdm import tqdm
import networkx as nx
import retworkx as rx
import osmnx as ox
import pandas as pd

base_speed = 21.0  # km/h                #path
map_filter = '["highway"]["area"!~"yes"]["access"!~"private"]["highway"!~"motorway|trunk|track|primary|abandoned|' \
             'bus_guideway|construction|corridor|elevator|escalator|footway|motor|planned|platform|proposed|raceway|' \
             'steps|service"]["bicycle"!~"no"]["service"!~"private|driveway"]'


# More car-road-like
map_filter = '["highway"]["area"!~"yes"]["access"!~"private"]["highway"!~"motorway|trunk|track|path|abandoned|' \
             'bus_guideway|construction|corridor|elevator|escalator|footway|motor|planned|platform|proposed|raceway|' \
             'steps|service|cycleway"]["service"!~"private|driveway"]'

DEBUG = False


def convert_retworkx_to_networkx(graph):
    """Convert a retworkx PyGraph or PyDiGraph to a networkx graph."""
    edge_list = []
    for e in graph.edge_list():
        data = graph.get_all_edge_data(*e)
        for edge_data in data:
            edge_list.append((graph[e[0]], graph[e[1]], edge_data))

    if isinstance(graph, rx.PyGraph):
        if graph.multigraph:
            return nx.MultiGraph(edge_list)
        else:
            return nx.Graph(edge_list)
    else:
        if graph.multigraph:
            return nx.MultiDiGraph(edge_list)
        else:
            return nx.DiGraph(edge_list)


def most_central_node(graph):
    xs = []
    ys = []
    for node in graph.nodes():
        x = graph.nodes[node]["x"]
        y = graph.nodes[node]["y"]

        xs.append(x)
        ys.append(y)

    centroid = (sum(xs) / len(xs), sum(ys) / len(ys))
    min_dist = 999999999
    min_node = None

    for node in graph.nodes():
        x = graph.nodes[node]["x"]
        y = graph.nodes[node]["y"]

        d = math.sqrt(((x - centroid[0]) ** 2) + ((y - centroid[1]) ** 2))
        if d < min_dist:
            min_dist = d
            min_node = node

    return min_node


def get_start_node(graph, query):
    query = query.lower()

    for node in graph.nodes():
        data = graph.nodes[node]
        if "name" in data:
            name = data["name"].lower()
        else:
            name = " ".join(map(str, data.values())).lower()

        if query in name:
            return node

    for edge in graph.edges():
        data = graph[edge[0]][edge[1]]
        if "name" in data[0]:
            names = data[0]["name"]
            if isinstance(names, list):
                name = " ".join(names).lower()
            else:
                name = str(names).lower()
        else:
            name = " ".join(map(str, data[0].values())).lower()
        if query in name:
            return edge[0]

    return most_central_node(graph)


def get_shortest_paths_distances(graph, pairs, edge_weight_name):
    """Compute shortest distance between each pair of nodes in a graph.
    Return a dictionary keyed on node pairs (tuples)."""
    distances = {}
    for pair in tqdm(pairs):
        path = rx.dijkstra_shortest_paths(graph, pair[0], pair[1], weight_fn=lambda e: e[edge_weight_name])
        d = 0.0
        key = list(path.keys())[0]
        nodes = [n for n in path[key]]
        edges = list(zip(nodes[:-1], nodes[1:]))
        for e in edges:
            d += graph.get_edge_data(*e)[edge_weight_name]
        distances[pair] = d
    return distances


def create_complete_graph_rx(pair_weights, flip_weights=True):
    """
    Create a completely connected graph using a list of vertex pairs and the shortest path distances between them
    Parameters:
        pair_weights: list[tuple] from the output of get_shortest_paths_distances
        flip_weights: Boolean. Should we negate the edge attribute in pair_weights?
    """
    g = rx.PyGraph()
    edge_list = []
    added_nodes = dict()
    for k, v in tqdm(pair_weights.items()):
        wt_i = - v if flip_weights else v

        if k[0] not in added_nodes:
            v1 = g.add_node(k[0])
            added_nodes[k[0]] = v1
        else:
            v1 = added_nodes[k[0]]

        if k[1] not in added_nodes:
            v2 = g.add_node(k[1])
            added_nodes[k[1]] = v2
        else:
            v2 = added_nodes[k[1]]

        edge = (v1, v2, {'length': v, 'weight': wt_i})
        edge_list.append(edge)

    g.extend_from_weighted_edge_list(edge_list)
    node_mapping = {val: key for key, val in added_nodes.items()}
    return g, node_mapping


def add_augmenting_path_to_graph(graph, min_weight_pairs, node_mapping):
    """
    Add the min weight matching edges to the original graph
    Parameters:
        graph: NetworkX graph (original graph from trailmap)
        min_weight_pairs: list[tuples] of node pairs from min weight matching
    Returns:
        augmented NetworkX graph
    """

    # We need to make the augmented graph a MultiGraph, so we can add parallel edges
    graph_aug = graph.copy()
    edge_list = []
    for pair in min_weight_pairs:
        v1 = node_mapping[pair[0]]
        v2 = node_mapping[pair[1]]

        path = rx.dijkstra_shortest_paths(graph, v1, v2, weight_fn=lambda e: e["length"])
        d = 0.0
        key = list(path.keys())[0]
        nodes = [n for n in path[key]]
        edges = list(zip(nodes[:-1], nodes[1:]))
        for e in edges:
            d += graph.get_edge_data(*e)["length"]

        edge = (v1, v2, {'length': d, 'trail': 'augmented'})
        edge_list.append(edge)

    graph_aug.extend_from_weighted_edge_list(edge_list)
    return graph_aug


def create_eulerian_circuit(graph_augmented, graph_original, starting_node=None):
    """Create the eulerian path using only edges from the original graph."""
    euler_circuit = []
    naive_circuit = list(nx.eulerian_path(graph_augmented, source=starting_node))
    
    for edge in naive_circuit:
        edge_data = graph_augmented.get_edge_data(edge[0], edge[1])    
        
        if not "trail" in edge_data[0]:  # or edge_data[0]['trail'] != 'augmented':
            # If `edge` exists in original graph, grab the edge attributes and add to eulerian circuit.
            edge_att = graph_original[edge[0]][edge[1]]
            euler_circuit.append((edge[0], edge[1], edge_att)) 
        else: 
            aug_path = nx.shortest_path(graph_original, edge[0], edge[1], weight='length')
            aug_path_pairs = list(zip(aug_path[:-1], aug_path[1:]))
            
            print('Filling in edges for augmented edge: {}'.format(edge))
            print('Augmenting path: {}'.format(' => '.join(map(str, aug_path))))
            print('Augmenting path pairs: {}\n'.format(aug_path_pairs))
            
            # If `edge` does not exist in original graph, find the shortest path between its nodes and 
            #  add the edge attributes for each link in the shortest path.
            for edge_aug in aug_path_pairs:
                edge_aug_att = graph_original[edge_aug[0]][edge_aug[1]]
                euler_circuit.append((edge_aug[0], edge_aug[1], edge_aug_att))
                                      
    return euler_circuit


def get_graph(boundary):
    if DEBUG:
        G = ox.graph_from_place("Messel, Germany",
                                retain_all=False,
                                truncate_by_edge=True,
                                custom_filter=map_filter,
                                clean_periphery=False,
                                simplify=False)

    else:
        # Load city boundaries manually created by Google MyMaps
        of_boundaries = json.load(open(boundary, "r"))
        poly = Polygon(of_boundaries)
        G = ox.graph_from_polygon(poly,
                                  retain_all=False,
                                  truncate_by_edge=True,
                                  custom_filter=map_filter,
                                  clean_periphery=False,
                                  simplify=False)

    G = ox.speed.add_edge_speeds(G)
    G = ox.speed.add_edge_travel_times(G)
    G = ox.utils_graph.get_undirected(G)

    # remove selfloops
    for self_loop_edge in list(nx.selfloop_edges(G)):
        G.remove_edge(*self_loop_edge)

    return G


def plot_graph():
    ec = ox.plot.get_edge_colors_by_attr(G, "speed_kph", cmap="inferno")
    fig, ax = ox.plot_graph(G, edge_color=ec, edge_linewidth=2, node_size=0)


def restore_node_attributes(graph, base_graph):
    for node_id in graph.nodes():
        if node_id in base_graph.nodes:
            node_data = base_graph.nodes[node_id]
            for key, val in node_data.items():
                graph.nodes[node_id][key] = val


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("%s <map.xml> <boundaries.json>" % sys.argv[0])
        sys.exit(-1)

    map_f = sys.argv[1]
    boundary_f = sys.argv[2]
    G = get_graph(boundary_f)

    start_node = get_start_node(G, query="Wilhelmsplatz")

    base_graph = G.copy()
    G = rx.networkx_converter(G)
    nodes_odd_degree = [idx for idx in G.node_indexes() if G.degree(idx) % 2 == 1]

    print('Number of nodes of odd degree: {}'.format(len(nodes_odd_degree)))
    print('Number of total nodes: {}'.format(len(G.nodes())))

    odd_node_pairs = list(itertools.combinations(nodes_odd_degree, 2))
    odd_node_pairs_shortest_paths = get_shortest_paths_distances(G, odd_node_pairs, 'length')
    g_odd_complete, node_mapping = create_complete_graph_rx(odd_node_pairs_shortest_paths, flip_weights=True)

    # Counts
    print('Number of nodes: {}'.format(len(g_odd_complete.nodes())))
    print('Number of edges: {}'.format(len(g_odd_complete.edges())))

    # Compute min weight matching.
    # Note: max_weight_matching uses the 'weight' attribute by default as the attribute to maximize.
    odd_matching_dupes = rx.max_weight_matching(g_odd_complete,
                                                max_cardinality=True,
                                                weight_fn=lambda e: int(e["weight"] * 1000))

    print('Number of edges in matching: {}'.format(len(odd_matching_dupes)))

    # Convert matching to list of deduped tuples
    odd_matching = list(pd.unique([tuple(sorted([k, v])) for k, v in odd_matching_dupes]))

    # Counts
    print('Number of edges in matching (deduped): {}'.format(len(odd_matching)))

    # Create augmented graph: add the min weight matching edges to g
    g_aug = add_augmenting_path_to_graph(G, odd_matching, node_mapping)

    # Counts
    print('Number of edges in original graph: {}'.format(len(G.edges())))
    print('Number of edges in augmented graph: {}'.format(len(g_aug.edges())))
    print(pd.value_counts([g_aug.degree(n) for n in g_aug.node_indexes()]))

    g_aug = convert_retworkx_to_networkx(g_aug)
    G = convert_retworkx_to_networkx(G)

    restore_node_attributes(g_aug, base_graph)
    restore_node_attributes(G, base_graph)

    # Create the Eulerian circuit
    euler_path = create_eulerian_circuit(g_aug, G, starting_node=start_node)

    # Preview first 20 directions of CPP solution
    for i, edge in enumerate(euler_path[0:20]):
        print(i, edge)

    # Computing some stats
    total_mileage_of_circuit = sum([edge[2][1]['length']
                                    if 1 in edge[2]
                                    else edge[2][0]['length']
                                    for edge in euler_path])
    total_mileage_on_orig_trail_map = sum(nx.get_edge_attributes(G, 'length').values())
    _vcn = pd.value_counts(pd.value_counts([(e[0]) for e in euler_path]), sort=False)
    node_visits = pd.DataFrame({'n_visits': _vcn.index, 'n_nodes': _vcn.values})
    _vce = pd.value_counts(pd.value_counts([sorted(e)[0] + sorted(e)[1] for e in nx.MultiDiGraph(euler_path).edges()]))
    edge_visits = pd.DataFrame({'n_visits': _vce.index, 'n_edges': _vce.values})

    # Printing stats
    print('Mileage of circuit: {0:.2f}'.format(total_mileage_of_circuit))
    print('Mileage on original trail map: {0:.2f}'.format(total_mileage_on_orig_trail_map))
    print('Mileage retracing edges: {0:.2f}'.format(total_mileage_of_circuit - total_mileage_on_orig_trail_map))
    print('Percent of mileage retraced: {0:.2f}%\n'.format(
        (1 - total_mileage_of_circuit / total_mileage_on_orig_trail_map) * -100))

    print('Number of edges in circuit: {}'.format(len(euler_path)))
    print('Number of edges in original graph: {}'.format(len(G.edges())))
    print('Number of nodes in original graph: {}\n'.format(len(G.nodes())))

    print('Number of edges traversed more than once: {}\n'.format(len(euler_path) - len(G.edges())))

    print('Number of times visiting each node:')
    print(node_visits.to_string(index=False))

    print('\nNumber of times visiting each edge:')

    print(edge_visits.to_string(index=False))

    current_dt = datetime(2022, 4, 1, 8, 0, 0, 0)
    with open("waypoints.txt", "w") as wpf:
        # wpf.write(f"name,desc,latitude,longitude\n")
        wpf.write(f"latitude,longitude,speed,date\n")
        for path in euler_path:
            from_vertex, to_vertex, edge_data = path

            from_lat = G.nodes[from_vertex]["y"]
            from_lon = G.nodes[from_vertex]["x"]
            to_lat = G.nodes[to_vertex]["y"]
            to_lon = G.nodes[to_vertex]["x"]
            name = edge_data[0].get("name", "Path")
            desc = edge_data[0].get("name", "")

            #wpf.write(f"{name},{desc},{from_lat},{from_lon}\n")
            wpf.write(f"{from_lat},{from_lon},{base_speed},{current_dt.isoformat()}\n")

            distance_m = edge_data[0]["length"]
            duration_s = distance_m / (base_speed / 3.6)

            current_dt += timedelta(seconds=duration_s)
