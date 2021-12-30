import math
import random

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

"""
 Use kMeans++ algorithm to determine initial centroids for kMeans, 
 based on pseudocode from lecture.
 :param P: Set of points like [[x, y], [x, y], [x, y]]
 :param k: the number of centroids
"""
def k_means_plus_plus(P, k):
    n = len(P)
    x = random.random()
    centroids = []
    centroids.append(math.ceil(x * n))
    minDistances = []
    for j in range(n):
        minDistances.append(float('inf'))
    for i in range(1, k):
        for j in range(n):
            x = distance(P[j], P[centroids[i-1]])
            if (minDistances[j] > x):
                minDistances[j] = x
        cumulative = []
        cumulative.append(minDistances[0] ** 2)
        for j in range(1, n):
            cumulative.append(cumulative[j-1] + minDistances[j] ** 2)
        x = random.random()
        x = x * cumulative[n-1]
        if (x <= cumulative[0]):
            index = 1
        else:
            for j in range(1, n):
                if (x > cumulative[j-1] and x <= cumulative[j]):
                    index = j
        centroids.append(index)
    return centroids

"""
 Use kMeans algorithm to cluster points, 
 based on pseudocode from lecture.
 :param P: Set of points like [[x, y], [x, y], [x, y]]
 :param k: the number of centroids
"""
def k_means(P, k):
    centroids = k_means_plus_plus(P, k)
    


def coreset_construction(P, k, eps):
    n = len(P)
    S = []
    a = 1
    z = math.log(n) * math.log(a * math.log(n))
    C = k_means(P, k)