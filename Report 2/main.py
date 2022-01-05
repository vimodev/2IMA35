import math
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from pyspark import SparkConf, SparkContext

from helpers import get_clustering_data, create_distance_matrix
from coreset import coreset_construction, coreset_verify

def divide(vertices, t):
    subsets = [[] for i in range(t)]
    for i in range(len(vertices)):
        subsets[i % t].append(vertices[i])
    return subsets

def coreset(vertices, k, eps):
    n = len(vertices)
    # Set default weights to 1
    for i in range(n):
        vertices[i].append(1)
    return coreset_construction(vertices, k, eps)

def mpc_coreset(vertices, k, eps):
    sc = SparkContext("local", "App Name")
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
        inputs = []
        for i in range(0, len(Qs) - 1, 2):
            print("Handling pair...")
            left = Qs[i]
            right = Qs[i + 1]
            inputs.append([left, right])
            # union = left + right
            # union_coreset = coreset_construction(union, k, eps / (4 * level))
            # newQs.append(union_coreset)
        rddS = sc.parallelize(inputs).map(lambda x: (coreset_construction(x[0] + x[1], k, eps / (4 * level))))
        newQs = rddS.collect()
        # If odd number, re add last untreated subset
        if (len(Qs) % 2 == 1):
            newQs.append(Qs[len(Qs) - 1])
        level += 1
        Qs = newQs
    return Qs[0]


        

def main():
    k = 10
    eps = 0.1

    datasets = get_clustering_data(1000)
    for dataset in datasets[0:1]:
        timestamp = datetime.now()
        print("Start creating Distance Matrix...")
        dm, E, size, vertex_coordinates = create_distance_matrix(dataset[0][0])
        V = list(range(len(dm)))
        print("Size dataset: ", len(dm))
        print("Created distance matrix in: ", datetime.now() - timestamp)
        print("Start creating coreset")
        timestamp = datetime.now()
        #S = mpc_coreset(vertex_coordinates, k, eps)
        S = coreset(vertex_coordinates, k, eps)
        print(len(S))
        # Verify if its correct
        print("Verifying coreset on random centroids...")
        for i in range(100):
            centroids = random.sample(vertex_coordinates, k)
            if not (coreset_verify(vertex_coordinates, S, centroids, eps)):
                print("Invalid coreset! Does not adhere to the error bound!")



if __name__ == '__main__':
    main()