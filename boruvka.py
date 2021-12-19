import math
import random
from datetime import datetime

from pyspark import SparkConf, SparkContext

from component import create_component, remove_component
from prim import prim
from helpers import create_distance_matrix, get_clustering_data, plot_mst


def boruvka(dataset):
    dm, E, size, vertex_coordinates = create_distance_matrix(dataset)
    components = {}
    for vertex in range(len(dataset)):
        create_component(components, vertex)


    while len(components.keys()) >= 2:
        random_component = random.choice(list(components.values()))
        random_component.merge_with_best(components, dm)

    mst = list(list(components.values())[0].edges)
    return mst
    # print(components)


if __name__ == '__main__':
    datasets = get_clustering_data()
    for dataset in datasets:
        boruvka(dataset[0])