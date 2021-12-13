import math
import random
from datetime import datetime

from pyspark import SparkConf, SparkContext

from component import create_component, remove_component
from prim import prim
from helpers import create_distance_matrix, get_clustering_data, plot_mst


def boruvka(dataset):
    dm, E, size, vertex_coordinates = create_distance_matrix(dataset[0][0])
    components = {}
    for vertex in range(len(dataset[0][0])):
        create_component(components, vertex)

    print("Start creating MST...")
    timestamp = datetime.now()

    while len(components.keys()) >= 2:
        random_component = random.choice(list(components.values()))
        random_component.merge_with_best(components, dm)

    print("Found MST in: ", datetime.now() - timestamp)

    mst = list(list(components.values())[0].edges)
    plot_mst(dataset[0][0], mst, False, False)

    # print(components)


def boruvka_mrc(dataset):
    conf = SparkConf().setAppName('MST_Algorithm').set("spark.driver.bindAddress", "127.0.0.1").setMaster(
        "local[4]").set("spark.yarn.appMasterEnv.JAVA_HOME", "/usr/lib/jvm/java-8-openjdk")
    global sc
    sc = SparkContext(conf=conf, pyFiles=['./component.py'])

    dm, E, size, vertex_coordinates = create_distance_matrix(dataset[0][0])
    components = {}
    for vertex in range(len(dataset[0][0])):
        create_component(components, vertex)
    while len(components.keys()) >= 2:
        mapping = find_best_neighbours_mpc(components, dm)
        contraction_mpc(components, dm, mapping)

    mst = list(list(components.values())[0].edges)
    plot_mst(dataset[0][0], mst, False, False)


def find_best_neighbours_mpc(p_components, p_dm):
    global sc

    leader_mapping = {}
    for components in p_components:
        for vertex in p_components[components].vertices:
            leader_mapping[vertex] = components
    print(leader_mapping)

    best_neighbours = sc.parallelize(list(p_components.values())).map(lambda x: (x.get_cheapest_neighbour(leader_mapping, p_dm)))

    best_neighbours = best_neighbours.collect()
    mapping = {}
    for edge in best_neighbours:
        mapping[edge[0]] = edge[1]

    return mapping


def contract(v, p_components, p_dm, mapping):
    S = []
    while v not in S:
        S.append(v)
        v = mapping[v]
    return min(S), S


def keep_largest(ar, arr):
    if len(ar) > len(arr):
        return ar
    return arr


def create_paths(a):
    leader_index = a[1].index(a[0])
    if leader_index == len(a[1]) - 1:
        return [a[1][::-1]]
    if leader_index > 0:
        return [[a[0]] + a[1][:leader_index][::-1], a[1][leader_index:]]
    return [a[1]]


def contraction_mpc(p_components, p_dm, mapping):
    global sc
    deletions = sc.parallelize(list(p_components.values())).map(
        lambda x: contract(x.leader, p_components, p_dm, mapping)).reduceByKey(lambda a, b: keep_largest(a, b))
    chains = deletions.collect()
    chains = sc.parallelize(chains).map(lambda a: (a[0], create_paths(a))).collect()
    merged = sc.parallelize(chains).map(lambda a: p_components[a[0]].merge_chains(a[1], p_components, p_dm))
    merged = merged.collect()

    for returns in merged:
        leader = returns[0]
        if leader in p_components:
            component = p_components[leader]
            component.vertices = set(returns[1][1])
            component.edges = set(returns[1][2])
            for to_remove in returns[1][0]:
                if to_remove != leader:
                    remove_component(p_components, to_remove)


if __name__ == '__main__':
    datasets = get_clustering_data()
    boruvka_mrc(datasets[0])
    # for dataset in datasets:
    #     boruvka_mrc(dataset)
