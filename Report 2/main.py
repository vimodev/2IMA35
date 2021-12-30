import math
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from pyspark import SparkConf, SparkContext

from helpers import get_clustering_data, create_distance_matrix
from coreset import coreset_construction

def main():
    datasets = get_clustering_data(2000)
    for dataset in datasets:
        timestamp = datetime.now()
        print("Start creating Distance Matrix...")
        dm, E, size, vertex_coordinates = create_distance_matrix(dataset[0][0])
        V = list(range(len(dm)))
        print("Size dataset: ", len(dm))
        print("Created distance matrix in: ", datetime.now() - timestamp)
        print("Start creating coreset")
        timestamp = datetime.now()
        coreset_construction(vertex_coordinates, 5, 0.02)



if __name__ == '__main__':
    main()