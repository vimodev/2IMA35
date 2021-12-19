import math
import random
from argparse import ArgumentParser
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from pyspark import SparkConf, SparkContext

from collections import deque

# Snap stanford
from boruvka import boruvka
from helpers import get_clustering_data, create_distance_matrix, get_key, plot_mst, plot_clustering


def partion_vertices(vertices, k):
    """
    Partitioning of the vertices in k smaller subsets (creates a partitioning twice
    :param vertices: all vertices
    :param k: number of subsets that need to be created
    :return: the partitioning in list format
    """
    U = []
    V = []
    random.shuffle(vertices)
    verticesU = vertices.copy()
    random.shuffle(vertices)
    verticesV = vertices.copy()
    for i in range(len(vertices)):
        if i < k:
            U.append({verticesU[i]})
            V.append({verticesV[i]})
        else:
            U[i % k].add(verticesU[i])
            V[i % k].add(verticesV[i])
    return U, V


def partition_edges(E, x):
    """
    Partition the edge set E into x smaller subsets
    :param E: All edges
    :param x: number of subsets that need to be created
    :return E divided into x subsets
    """
    S = [set() for i in range(math.ceil(x))]
    step = 1 / math.ceil(x)
    # Go over all edges
    for i in range(len(E)):
        for j in E[i]:
            # Hash it to a partition
            bucket = math.floor(random.random() / step)
            # And add the edge to it
            S[bucket].add((i, j, E[i][j]))
    # Change back from set partitions to list partitions
    # Because thats what the other guy also did
    L = [[] for i in range(math.ceil(x))]
    for i in range(math.ceil(x)):
        L[i] = list(S[i])
    return L


def find_mst(U, V, E):
    """
    finds the mst of graph G = (U union V, E)
    :param U: vertices U
    :param V: vertices V
    :param E: edges of the graph
    :return: the mst and edges not in the mst of the graph
    """
    vertices = set()
    for v in V:
        vertices.add(v)
    for u in U:
        vertices.add(u)
    E = sorted(E, key=get_key)
    connected_component = set()
    mst = []
    remove_edges = set()
    while len(mst) < len(vertices) - 1 and len(connected_component) < len(vertices):
        if len(E) == 0:
            break
        change = False
        i = 0
        while i < len(E):
            if len(connected_component) == 0:
                connected_component.add(E[i][0])
                connected_component.add(E[i][1])
                mst.append(E[i])
                change = True
                E.remove(E[i])
                break
            else:
                if E[i][0] in connected_component:
                    if E[i][1] in connected_component:
                        remove_edges.add(E[i])
                        E.remove(E[i])
                    else:
                        connected_component.add(E[i][1])
                        mst.append(E[i])
                        E.remove(E[i])
                        change = True
                        break
                elif E[i][1] in connected_component:
                    if E[i][0] in connected_component:
                        remove_edges.add(E[i])
                        E.remove(E[i])
                    else:
                        connected_component.add(E[i][0])
                        mst.append(E[i])
                        E.remove(E[i])
                        change = True
                        break
                else:
                    i += 1
        if not change:
            if len(E) != 0:
                connected_component.add(E[0][0])
                connected_component.add(E[0][1])
                mst.append(E[0])
                E.remove(E[0])
    for edge in E:
        remove_edges.add(edge)
    # if len(mst) != len(vertices) - 1 or len(connected_component) != len(vertices):
    # print("Warning: parition cannot have a full MST! Missing edges to create full MST.")
    # print("Error: MST found cannot be correct \n Length mst: ", len(mst), "\n Total connected vertices: ",
    #       len(connected_component), "\n Number of vertices: ", len(vertices))
    return mst, remove_edges


def get_edges(U, V, E):
    """
    :param U: subset of vertices (u_j)
    :param V: subset of vertices (v_i)
    :param E: all edges of the whole graph
    :return: all edges that are part of the graph u_j U v_j
    """
    edges = set()
    for node1 in U:
        for node2 in V:
            if node2 in E[node1]:
                edges.add((node1, node2, E[node1][node2]))
            elif node1 in E[node2]:
                edges.add((node2, node1, E[node2][node1]))
    edge_list = []
    for edge in edges:
        edge_list.append(edge)
    return U, V, edge_list


def total_edges(E):
    """
    Compute total number of edges, to be sure
    :param E: the edge set
    :return |E|
    """
    c = 0
    for i in range(len(E)):
        c += len(E[i])
    return c / 2


def reduce_edges(vertices, E, c, epsilon):
    """
    Uses PySpark to distribute the computation of the MSTs,
    Randomly partition the vertices twice in k subsets (U = {u_1, u_2, .., u_k}, V = {v_1, v_2, .., v_k})
    For every intersection between U_i and V_j, create the subgraph and find the MST in this graph
    Remove all edges from E that are not part of the MST in the subgraph
    :param vertices: vertices in the graph
    :param E: edges of the graph
    :param c: constant
    :param epsilon:
    :return:The reduced number of edges
    """
    SparkContext.setSystemProperty('spark.executor.memory', '10g')
    sc = SparkContext("local", "App Name")
    n = len(vertices)
    # k = math.ceil(n ** ((c - epsilon) / 2))
    # U, V = partion_vertices(vertices, k)

    # We partition all edges into |E| / n^1+eps partitions
    S = partition_edges(E, total_edges(E) / (n ** (1 + epsilon)))
    print(total_edges(E) / (n ** (1 + epsilon)))
    global total_size
    percentage_graph.append(len(S[0]) / total_size)
    print("SEND TO MACHINE SIZE, ", len(S[0]))
    # We send the subgraph G=(V, E_i) to each machine
    rddS = sc.parallelize(S).map(lambda x: (find_mst(vertices, [], x)))
    both = rddS.collect()
    # rddUV = sc.parallelize(U).cartesian(sc.parallelize(V)).map(lambda x: get_edges(x[0], x[1], E)).map(
    #     lambda x: (find_mst(x[0], x[1], x[2])))
    # both = rddUV.collect()

    mst = []
    removed_edges = set()
    for i in range(len(both)):
        mst.append(both[i][0])
        for edge in both[i][1]:
            removed_edges.add(edge)

    sc.stop()
    return mst, removed_edges


def remove_edges(E, removed_edges):
    """
    Removes the edges, which are removed when generating msts
    :param E: current edges
    :param removed_edges: edges to be removed
    :param msts: edges in the msts
    :return: return the updated edge dict
    """
    for edge in removed_edges:
        if edge[1] in E[edge[0]]:
            del E[edge[0]][edge[1]]
    return E

total_size = 0
percentage_graph = []
def create_mst(V, E, epsilon, size, vertex_coordinates, plot_itermediate=False):
    """
    Creates the mst of the graph G = (V, E).
    As long as the number of edges is greater than n ^(1 + epsilon), the number of edges is reduced
    Then the edges that needs to be removed are removed from E and the size is updated.
    :param V: Vertices
    :param E: edges
    :param epsilon:
    :param m: number of machines
    :param size: number of edges
    :return: returns the reduced graph with at most np.power(n, 1 + epsilon) edges
    """
    rounds = 0
    n = len(V)
    c = math.log(size / n, n)
    global total_size
    total_size = size
    while size > np.power(n, 1 + epsilon):
        rounds += 1
        print("C: ", c)
        mst, removed_edges = reduce_edges(V, E, c, epsilon)
        if plot_itermediate:
            plot_mst(vertex_coordinates, mst, True, False)
        E = remove_edges(E, removed_edges)
        print("Total edges removed in this iteration", len(removed_edges))
        size = size - len(removed_edges)
        print("New total of edges: ", size)
        c = (c - epsilon) / 2
    # Now the number of edges is reduced and can be moved to a single machine
    V = set(range(n))
    items = E.items()  # returns [(x, {y : 1})]
    edges = []
    for item in items:
        items2 = item[1].items()
        for item2 in items2:
            edges.append((item[0], item2[0], item2[1]))
    mst, removed_edges = find_mst(V, V, edges)
    from benchmarker import used_rounds
    used_rounds.append(rounds)
    plt.plot(percentage_graph)
    plt.show()
    print(f"ROUND OF COMMUNICATIONS FOR, {len(V)}, {rounds}")
    return mst


def prim(dataset):
    vertex_count = len(dataset[0][0])
    dm, E, size, vertex_coordinates = create_distance_matrix(dataset[0][0])

    used = set()
    unused = set()
    start = random.randint(0, vertex_count)
    used.add(start)

    mst = []
    for i in range(0, vertex_count):
        if i != start:
            unused.add(i)

    print("[PRIM] Start creating MST...")
    timestamp = datetime.now()
    while len(unused) != 0:
        if len(unused) % 100 == 0:
            print(len(used), vertex_count)
        min_weight = math.inf
        s = -1
        t = -1
        for vertex in used:
            for target in unused:
                cost = dm[vertex][target]
                if min_weight > cost:
                    min_weight = cost
                    s = vertex
                    t = target
        mst.append((s, t, dm[s][t]))
        unused.remove(t)
        used.add(t)
    print("[PRIM] Found MST in: ", datetime.now() - timestamp)
    score = 0
    for edge in mst:
        score = score + edge[2]
    print("[PRIM] Score: ", score)
    plot_mst(dataset[0][0], mst, False, False)


def create_adjacency_list(n, mst):
    adj = [[] for i in range(n)]
    for edge in mst:
        adj[edge[0]].append(edge[1])
        adj[edge[1]].append(edge[0])
    return adj


def get_edge_cluster(clusters, e):
    for c in clusters:
        if e[0] in c and e[1] in c:
            return clusters.index(c)


def create_clustering(n, mst, k):
    # Sort mst to get highest weighted edges
    mst.sort(key=lambda x: x[2], reverse=True)
    # Create adj matrix for ez BFS
    adj = create_adjacency_list(n, mst)
    # Initialize clusters data structure
    clusters = []
    # Initially everything in 1 cluster
    clusters.append(list(range(n)))
    if (k == 1):
        return clusters
    # Split clusters on longest edge
    for c in range(k - 1):
        # Remove the longest edge
        rm = mst[c]
        hc = get_edge_cluster(clusters, rm)
        adj[rm[0]].remove(rm[1])
        adj[rm[1]].remove(rm[0])
        clusters.append([])
        # Perform BFS on one of the edge's nodes
        Q = deque()
        explored = [False for i in range(n)]
        explored[rm[0]] = True
        Q.append(rm[0])
        while (len(Q) != 0):
            v = Q.popleft()
            clusters[c + 1].append(v)
            clusters[hc].remove(v)
            for trg in adj[v]:
                if (explored[trg]):
                    continue
                explored[trg] = True
                Q.append(trg)
    return clusters


def main():
    """
    For every dataset, it creates the mst and plots the clustering
    """
    parser = ArgumentParser()
    parser.add_argument('--test', help="Used for smaller dataset and testing", action="store_true")
    parser.add_argument('--epsilon', help="epsilon [default=1/8]", type=float, default=1 / 8)
    parser.add_argument('--machines', help="Number of machines [default=1]", type=int, default=10)
    args = parser.parse_args()

    print("Start generating MST")
    if args.test:
        print("Test argument given")

    start_time = datetime.now()
    print("Starting time:", start_time)

    # create_mst()
    datasets = get_clustering_data(2000)
    for dataset in datasets:
        timestamp = datetime.now()
        print("Start creating Distance Matrix...")
        dm, E, size, vertex_coordinates = create_distance_matrix(dataset[0][0])
        V = list(range(len(dm)))
        print("Size dataset: ", len(dm))
        print("Created distance matrix in: ", datetime.now() - timestamp)
        print("Start creating MST...")
        timestamp = datetime.now()
        mst = create_mst(V, E, epsilon=args.epsilon, size=size, vertex_coordinates=vertex_coordinates)
        print("Found MST in: ", datetime.now() - timestamp)
        print("Start creating plot of MST...")
        timestamp = datetime.now()

        plot_mst(dataset[0][0], mst, False, False)
        k = 2
        clusters = create_clustering(len(V), mst, k)
        plot_clustering(dataset[0][0], clusters)
        print("Created plot of MST in: ", datetime.now() - timestamp)

    print("Done...")


def mst_create(dataset):
    global percentage_graph
    percentage_graph = []
    print("Start creating MST...")
    timestamp = datetime.now()
    dm, E, size, vertex_coordinates = create_distance_matrix(dataset)
    V = list(range(len(dm)))
    mst = create_mst(V, E, epsilon=(1/8), size=size, vertex_coordinates=vertex_coordinates)
    print("Found MST in: ", datetime.now() - timestamp)
    return mst


if __name__ == '__main__':
    main()
