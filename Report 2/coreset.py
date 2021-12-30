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
    centroids.append(P[math.ceil(x * n)])
    minDistances = []
    for j in range(n):
        minDistances.append(float('inf'))
    for i in range(1, k):
        for j in range(n):
            x = distance(P[j], centroids[i-1])
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
        centroids.append(P[index])
    return centroids

"""
 Given the centroids, compute the clustering
"""
def compute_labels(P, centroids):
    labels = [-1 for i in range(len(P))]
    for i in range(len(P)):
        minDistance = float('inf')
        for j in range(len(centroids)):
            d = distance(P[i], centroids[j])
            if (d < minDistance):
                minDistance = d
                labels[i] = j
    return labels

def compute_centroids(P, labels, k):
    cluster_sizes = [0 for i in range(k)]
    point_sum = [[0, 0] for i in range(k)]
    for i in range(len(P)):
        cluster_sizes[labels[i]] += 1
        point_sum[labels[i]][0] += P[i][0]
        point_sum[labels[i]][1] += P[i][1]
    centroids = [[0,0] for i in range(k)]
    for i in range(k):
        centroids[i][0] = point_sum[i][0] / cluster_sizes[i]
        centroids[i][1] = point_sum[i][1] / cluster_sizes[i]
    return centroids

def centroids_changed(old, new):
    for i in range(len(old)):
        if (old[i][0] != new[i][0] or old[i][1] != new[i][1]):
            return True
    return False

"""
 Use kMeans algorithm to cluster points, 
 based on pseudocode from lecture.
 :param P: Set of points like [[x, y], [x, y], [x, y]]
 :param k: the number of centroids
"""
def k_means(P, k):
    centroids = k_means_plus_plus(P, k)
    changed = True
    while changed:
        print(centroids)
        labels = compute_labels(P, centroids)
        old_centroids = centroids
        centroids = compute_centroids(P, labels, k)
        changed = centroids_changed(old_centroids, centroids)
    return centroids, compute_labels(P, centroids)


def cost(P, centroids, labels):
    sum = 0
    for i in range(len(P)):
        sum += distance(P[i], centroids[labels[i]])
    return sum


def coreset_construction(P, k, eps):
    n = len(P)
    S = []
    a = 1
    z = math.log(n) * math.log(a * math.log(n))
    C, labels = k_means(P, k)
    r = math.sqrt(cost(P, C, labels) / (a * math.log(n) * n))