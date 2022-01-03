import math
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from pyspark import SparkConf, SparkContext

from helpers import get_clustering_data, create_distance_matrix
from coreset import coreset_construction

def divide(vertices, t):
    subsets = [[] for i in range(t)]
    for i in range(len(vertices)):
        subsets[i % t].append(vertices[i])
    return subsets

def mpc_coreset(vertices, k, eps):
    n = len(vertices)
    # Set default weights to 1
    for i in range(n):
        vertices[i].append(1)

    t = math.ceil(math.sqrt(n))
    Ps = divide(vertices, t)
    Qs = [coreset_construction(Ps[i], k, eps) for i in range(len(Ps))]
    level = 2
    # Simple while loop for now, this must become pyspark parallelized
    while (len(Qs) != 1):
        print("Level: " + str(level))
        print("#Qs: " + str(len(Qs)))
        # Combine all pairs, and compute coreset of coreset
        newQs = []
        for i in range(0, len(Qs) - 1, 2):
            print("Handling pair...")
            left = Qs[i]
            right = Qs[i + 1]
            union = left + right
            union_coreset = coreset_construction(union, k, eps / (4 * level))
            newQs.append(union_coreset)
        # If odd number, re add last untreated subset
        if (len(Qs) % 2 == 1):
            newQs.append(Qs[len(Qs) - 1])
        level += 1
        Qs = newQs
    return Qs[0]


        

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
        S = mpc_coreset(vertex_coordinates, 5, 0.1)
        print(len(S))



if __name__ == '__main__':
    main()