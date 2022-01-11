import math
import random
from datetime import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
from pyspark import SparkConf, SparkContext

from helpers import get_clustering_data, create_vertex_coordinates
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
    print("Dividing vertices into " + str(t) + " subsets...")
    Ps = divide(vertices, t)
    print("Computing coresets...")
    rddS = sc.parallelize(Ps).map(lambda P: (coreset_construction(P, k, eps)))
    Qs = rddS.collect()
    level = 2
    print("Starting coreset merging...")
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
    ks = [10]
    eps = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1]

    sizes = [1000, 5000, 10000, 50000, 100000]
    size = sizes[4]
    datasets = get_clustering_data(size, 0.06)

    for k in ks:
        print("k ", k)
        coresets = [[] for i in range(len(datasets))]
        for d in range(len(datasets)):
            print("Dataset ", d)
            dataset = datasets[d]
            vertices = create_vertex_coordinates(dataset[0][0])
            for e in eps:
                print("Epsilon ", e)
                coresets[d].append(coreset(vertices, k, e))

        coreset_sizes = [[len(coresets[d][e]) for e in range(len(eps))] for d in range(len(datasets))]

        plt.clf()
        plt.plot(eps, coreset_sizes[0], label="circles")
        plt.plot(eps, coreset_sizes[1], label="moons")
        plt.plot(eps, coreset_sizes[2], label="blobs")
        plt.legend()
        plt.xticks(eps, eps, fontsize=8)
        plt.savefig(str(size) + '-sizes-' + str(k) + '.png', bbox_inches='tight')

    # datasets = get_clustering_data(10000, 0.06)
    # for dataset in datasets:
    #     timestamp = datetime.now()
    #     vertex_coordinates = create_vertex_coordinates(dataset[0][0])
    #     V = list(range(len(vertex_coordinates)))
    #     print("Size dataset: ", len(vertex_coordinates))
    #     print("Start creating coreset")
    #     #S = mpc_coreset(vertex_coordinates, k, eps)
    #     S = coreset(vertex_coordinates, k, eps)
    #     print(len(S))
    #     print("Coreset took: ", datetime.now() - timestamp)
        # # Verify if its correct
        # print("Verifying coreset on random centroids...")
        # for i in range(100):
        #     centroids = random.sample(vertex_coordinates, k)
        #     if not (coreset_verify(vertex_coordinates, S, centroids, eps)):
        #         print("Invalid coreset! Does not adhere to the error bound!")



if __name__ == '__main__':
    main()