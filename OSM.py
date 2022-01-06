#!/usr/bin/env python
# coding: utf-8

"""

    D E P R E C A T E D

"""

import itertools
import json
from math import sin, cos, sqrt, atan2, radians
import sys
import os

from tqdm import tqdm
import networkx as nx
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

R = 6373.0
restriced_ways = ["track", "path"]


def plot_bound_point(point: tuple = None):
    x,y = zip(*of_boundaries)
    for i in range(0, len(x)):
        plt.plot(x[i:i+2], y[i:i+2], 'ro-')

    plt.plot(point[0], point[1], 'bo')
    
    plt.show()


def pip(p):
    poly = Polygon(of_boundaries)
    point = Point(p)
    plt.plot(*poly.exterior.xy)
    return point.within(poly)


def get_shortest_paths_distances(graph, pairs, edge_weight_name):
    """Compute shortest distance between each pair of nodes in a graph.
    Return a dictionary keyed on node pairs (tuples)."""
    distances = {}
    for pair in pairs:
        distances[pair] = nx.dijkstra_path_length(graph, pair[0], pair[1], weight=edge_weight_name)
    return distances


def create_complete_graph(pair_weights, flip_weights=True):
    """
    Create a completely connected graph using a list of vertex pairs and the shortest path distances between them
    Parameters: 
        pair_weights: list[tuple] from the output of get_shortest_paths_distances
        flip_weights: Boolean. Should we negate the edge attribute in pair_weights?
    """
    g = nx.Graph()
    for k, v in pair_weights.items():
        wt_i = - v if flip_weights else v
        # g.add_edge(k[0], k[1], {'distance': v, 'weight': wt_i})  # deprecated after NX 1.11 
        g.add_edge(k[0], k[1], **{'distance': v, 'weight': wt_i})  
    return g


def add_augmenting_path_to_graph(graph, min_weight_pairs):
    """
    Add the min weight matching edges to the original graph
    Parameters:
        graph: NetworkX graph (original graph from trailmap)
        min_weight_pairs: list[tuples] of node pairs from min weight matching
    Returns:
        augmented NetworkX graph
    """
    
    # We need to make the augmented graph a MultiGraph so we can add parallel edges
    graph_aug = nx.MultiGraph(graph.copy())
    for pair in min_weight_pairs:
        graph_aug.add_edge(pair[0], 
                           pair[1], 
                           **{'distance': nx.dijkstra_path_length(graph, pair[0], pair[1]), 'trail': 'augmented'}
                           # attr_dict={'distance': nx.dijkstra_path_length(graph, pair[0], pair[1]),
                           #            'trail': 'augmented'}  # deprecated after 1.11
                          )
    return graph_aug


def create_eulerian_circuit(graph_augmented, graph_original, starting_node=None):
    """Create the eulerian path using only edges from the original graph."""
    euler_circuit = []
    naive_circuit = list(nx.eulerian_path(graph_augmented, source=starting_node))
    
    for edge in naive_circuit:
        edge_data = graph_augmented.get_edge_data(edge[0], edge[1])    
        
        if not "trail" in edge_data[0]: # or edge_data[0]['trail'] != 'augmented':
            # If `edge` exists in original graph, grab the edge attributes and add to eulerian circuit.
            edge_att = graph_original[edge[0]][edge[1]]
            euler_circuit.append((edge[0], edge[1], edge_att)) 
        else: 
            aug_path = nx.shortest_path(graph_original, edge[0], edge[1], weight='distance')
            aug_path_pairs = list(zip(aug_path[:-1], aug_path[1:]))
            
            print('Filling in edges for augmented edge: {}'.format(edge))
            print('Augmenting path: {}'.format(' => '.join(aug_path)))
            print('Augmenting path pairs: {}\n'.format(aug_path_pairs))
            
            # If `edge` does not exist in original graph, find the shortest path between its nodes and 
            #  add the edge attributes for each link in the shortest path.
            for edge_aug in aug_path_pairs:
                edge_aug_att = graph_original[edge_aug[0]][edge_aug[1]]
                euler_circuit.append((edge_aug[0], edge_aug[1], edge_aug_att))
                                      
    return euler_circuit


def distance(lat1, lon1, lat2, lon2):
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("%s <map.xml> <boundaries.json>" % sys.argv[0])
        sys.exit(-1)

    map_f = sys.argv[1]
    boundary_f = sys.argv[2]

    # Load city boundaries manually created by Google MyMaps
    of_boundaries = json.load(open(boundary_f, "r"))

    poly = Polygon(of_boundaries)
    plt.plot(*poly.exterior.xy)
    plt.show()

    plot_bound_point([8.7369833, 50.1115023])

    tree = ET.parse(map_f)
    root = tree.getroot()

    nodes = dict()
    ways = set()
    relations = set()

    for child in tqdm(root):
        # print(child.tag, child.attrib)
        if child.tag == "node":
            nodes[child.get("id")] = child

        elif child.tag == "way":
            ways |= {child}

        elif child.tag == "relation":
            relations |= {child}

    print(f"Number of nodes: {len(nodes)}")
    print(f"Number of ways: {len(ways)}")
    print(f"Number of relations: {len(relations)}")

    paths = []
    checked_nodes = dict()

    for way in tqdm(ways):
        first_node = None
        last_node = None
        highway = None
        name = None

        tmp_paths = []

        for node in way:
            if node.tag == "nd":

                node_ref = node.get("ref")
                node_obj = nodes[node_ref]
                node_lat = float(node_obj.get("lat"))
                node_lon = float(node_obj.get("lon"))

                in_of = pip((node_lon, node_lat))
                if not in_of:
                    continue

                if node_ref not in checked_nodes:
                    checked_nodes[node_ref] = 1
                else:
                    checked_nodes[node_ref] += 1

                if first_node is None:
                    first_node = node
                    last_node = node
                    continue

                first_node = node

                if last_node is not first_node:
                    tmp_paths.append([first_node, last_node])

                last_node = node

            elif node.tag == "tag":
                if node.get("k") == "name":
                    name = node.get("v")

                elif node.get("k") == "highway":
                    highway = node.get("v")

                    if highway in restriced_ways:
                        highway = None

        if highway is not None and name is not None:
            for tmp_path in tmp_paths:
                first_node, last_node = tmp_path

                f_id = first_node.get("ref")
                l_id = last_node.get("ref")

                paths.append([first_node, last_node, name, highway])

    with open("paths.txt", "w") as pout:
        pout.write("start\tend\tlat\tlon\tlat\tlon\tname\n")

        with open("nodes.txt", "w") as nout:
            for path in paths:
                first_node, last_node, name, highway = path
                f_id = first_node.get("ref")
                l_id = last_node.get("ref")

                f_lat = float(nodes[f_id].get("lat")) * -100000
                f_lon = float(nodes[f_id].get("lon")) * 100000
                l_lat = float(nodes[l_id].get("lat")) * -100000
                l_lon = float(nodes[l_id].get("lon")) * 100000

                nout.write(f"{f_id};{f_lat};{f_lon}\n")
                nout.write(f"{l_id};{l_lat};{l_lon}\n")

                pout.write(f"{f_id}\t{l_id}\t{f_lat}\t{f_lon}\t{l_lat}\t{l_lon}\t{name}\n")

    G = nx.Graph()
    for path in paths:
        first_node, last_node, name, highway = path
        f_id = first_node.get("ref")
        l_id = last_node.get("ref")

        f_lat = float(nodes[f_id].get("lat"))
        f_lon = float(nodes[f_id].get("lon"))
        l_lat = float(nodes[l_id].get("lat"))
        l_lon = float(nodes[l_id].get("lon"))

        dist = distance(f_lat, f_lon, l_lat, l_lon)

        f_lat = f_lat * -100000
        f_lon = f_lon * 100000
        l_lat = l_lat * -100000
        l_lon = l_lon * 100000

        G.add_node(f_id, lat=f_lat, lon=f_lon)
        G.add_node(l_id, lat=l_lat, lon=l_lon)
        G.add_edge(f_id, l_id, name=name, distance=dist)

    G.number_of_edges()

    max_cc = max([len(l) for l in list(nx.connected_components(G))])
    print("Max Size Connected Component: ", max_cc)
    for cc in list(nx.connected_components(G)):
        size = len(cc)
        if not size == max_cc:
            for node in cc:
                G.remove_node(node)
            print(f"Removed {size}")

    found_2deg = True
    while found_2deg:
        found_2deg = False
        for node in G.nodes:
            neighbours = G[node]
            if len(neighbours) == 2:
                before, after = neighbours
                prev_edge = G[before][node]
                next_edge = G[node][after]

                prev_distance = prev_edge["distance"]
                next_distance = next_edge["distance"]
                new_distance = prev_distance + next_distance

                G.remove_edge(before, node)
                G.remove_edge(node, after)
                G.remove_node(node)
                G.add_edge(before, after, name=prev_edge["name"], distance=new_distance)
                found_2deg = True
                break

    with open("pathsx.txt", "w") as pout:
        pout.write("start\tend\tlat\tlon\tlat\tlon\tname\tdistance\n")
        for edge in G.edges:
            start, end = edge
            name = G[start][end]["name"]
            distance = G[start][end]["distance"]
            start_node = G.nodes[start]
            end_node = G.nodes[end]
            start_lat = start_node["lat"]
            start_lon = start_node["lon"]
            end_lat = end_node["lat"]
            end_lon = end_node["lon"]

            pout.write(f"{start}\t{end}\t{start_lat}\t{start_lon}\t{end_lat}\t{end_lon}\t{name}\t{distance}\n")

    sum(nx.get_edge_attributes(G, "distance").values())

    # https://brooksandrew.github.io/simpleblog/articles/intro-to-graph-optimization-solving-cpp/#intro-to-graph-optimization-with-networkx-in-python
    nodes_odd_degree = [v for v, d in G.degree() if d % 2 == 1]
    print('Number of nodes of odd degree: {}'.format(len(nodes_odd_degree)))
    print('Number of total nodes: {}'.format(len(G.nodes())))

    odd_node_pairs = list(itertools.combinations(nodes_odd_degree, 2))
    odd_node_pairs_shortest_paths = get_shortest_paths_distances(G, odd_node_pairs, 'distance')
    print(dict(list(odd_node_pairs_shortest_paths.items())[0:10]))

    g_odd_complete = create_complete_graph(odd_node_pairs_shortest_paths, flip_weights=True)

    # Counts
    print('Number of nodes: {}'.format(len(g_odd_complete.nodes())))
    print('Number of edges: {}'.format(len(g_odd_complete.edges())))

    # Compute min weight matching.
    # Note: max_weight_matching uses the 'weight' attribute by default as the attribute to maximize.
    odd_matching_dupes = nx.algorithms.max_weight_matching(g_odd_complete, True)

    print('Number of edges in matching: {}'.format(len(odd_matching_dupes)))

    # Convert matching to list of deduped tuples
    odd_matching = list(pd.unique([tuple(sorted([k, v])) for k, v in odd_matching_dupes]))

    # Counts
    print('Number of edges in matching (deduped): {}'.format(len(odd_matching)))

    # Preview of deduped matching
    print(odd_matching)

    # Create augmented graph: add the min weight matching edges to g
    g_aug = add_augmenting_path_to_graph(G, odd_matching)

    # Counts
    print('Number of edges in original graph: {}'.format(len(G.edges())))
    print('Number of edges in augmented graph: {}'.format(len(g_aug.edges())))
    print(pd.value_counts([e[1] for e in g_aug.degree()]))

    naive_euler_path = list(nx.eulerian_path(g_aug, source='36321196'))
    print('Length of eulerian path: {}'.format(len(naive_euler_path)))

    # Create the Eulerian circuit
    euler_path = create_eulerian_circuit(g_aug, G, '36321196')

    # Preview first 20 directions of CPP solution
    for i, edge in enumerate(euler_path[0:20]):
        print(i, edge)

    # Computing some stats
    total_mileage_of_circuit = sum([edge[2]['distance'] for edge in euler_path])
    total_mileage_on_orig_trail_map = sum(nx.get_edge_attributes(G, 'distance').values())
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

    euler_path_points = []
    with open("euler.txt", "w") as euf:
        for edge in euler_path:
            start = edge[0]
            end = edge[1]
            data = edge[2]

            start_node = G.nodes[start]
            end_node = G.nodes[end]

            start_lat = -1 * round(float(start_node["lat"]), 2) / 100000
            start_lon = round(float(start_node["lon"]), 2) / 100000.0
            end_lat = -1 * round(float(end_node["lat"]), 2) / 100000
            end_lon = round(float(end_node["lon"]), 2) / 100000.0

            euf.write(f"{start_lat}\t{start_lon}\n")

            name = data["name"]
            distance = data["distance"]

            euler_path_points.append([start_lat, start_lon, distance, name])

    print(len(euler_path_points))
    for i in range(len(euler_path_points) - 5, len(euler_path_points)):
        print(i)
        euler_path_points[i]

    BBox = (8.7286, 8.7963,
            50.0743, 50.1151)

    fig = px.scatter_mapbox(of_boundaries, lat=1, lon=0, zoom=12)
    fig.update_layout(mapbox_style="open-street-map")
    fig.show()

    folder = "images_2"
    if not os.path.exists(folder):
        os.mkdir(folder)

    distance = 0

    for i in range(2, len(euler_path_points)):
        print(i)
        points = euler_path_points[0:i]
        current_point = points[-1]

        distance += current_point[2]
        rounded_distance = '{:.3f}'.format(round(distance, 3)).zfill(7) + " km"

        print(rounded_distance)

        center = {"lat": current_point[0], "lon": current_point[1]}

        fig = px.line_mapbox(points, lat=0, lon=1, zoom=12)
        fig.update_layout(mapbox_style="open-street-map")

        # annotations=[
        #       go.layout.Annotation
        #   ]

        fig.add_annotation(
            text='Distanz:<br>' + rounded_distance,
            align='right',
            showarrow=False,
            xref='paper',
            yref='paper',
            x=1.0,
            y=1.0,
            bordercolor='red',
            borderwidth=2,
            bgcolor="#f1f1f1",
        )

        # fig.show()
        count = str(i).zfill(5)
        fig.write_image(folder + "/fig" + count + ".png")
        del fig

    print("ok")