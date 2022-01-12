import numpy as np
import scipy.spatial
from matplotlib import pyplot as plt
from sklearn.datasets import make_circles, make_moons, make_blobs


def get_clustering_data(n_samples=1500, noise=0):
    """
    Retrieves all toy datasets from sklearn
    :return: circles, moons, blobs datasets.
    """
    noisy_circles = make_circles(n_samples=n_samples, factor=.5,
                                 noise=noise)
    noisy_moons = make_moons(n_samples=n_samples, noise=noise)
    blobs = make_blobs(n_samples=n_samples, random_state=8)

    datasets = [
        (noisy_circles, {'damping': .77, 'preference': -240,
                         'quantile': .2, 'n_clusters': 2,
                         'min_samples': 20, 'xi': 0.25}),
        (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
        (blobs, {})]

    return datasets

def create_vertex_coordinates(dataset):
    vertices = []
    for line in dataset:
        vertices.append([line[0], line[1]])
    return vertices

def create_distance_matrix(dataset):
    """
    Creates the distance matrix for a dataset with only vertices. Also adds the edges to a dict.
    :param dataset: dataset without edges
    :return: distance matrix, a dict of all edges and the total number of edges
    """
    vertices = []
    size = 0
    for line in dataset:
        vertices.append([line[0], line[1]])
    d_matrix = scipy.spatial.distance_matrix(vertices, vertices, threshold=1000000)
    dict = {}
    # Run with less edges
    for i in range(len(d_matrix)):
        dict2 = {}
        for j in range(i, len(d_matrix)):
            if i != j:
                size += 1
                dict2[j] = d_matrix[i][j]
        dict[i] = dict2
    return d_matrix, dict, size, vertices


def load_txt_to_data(filepath):
    """
    Loads a CSV dataset
    """
    dataset = []
    with open(filepath) as file:
        for line in file:
            line = line.strip("\t")
            line = line.strip("\n")
            line = line.strip("\ufeff")
            line = line.split(",")
            line = [float(line[0]), float(line[1])]
            dataset.append(line)
    dataset = np.array(dataset)
    plt.scatter(x=dataset[:, 0], y=dataset[:, 1], s=1)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    return [dataset], {}


def get_key(item):
    """
    returns the sorting criteria for the edges. All edges are sorted from small to large values
    :param item: one item
    :return: returns the weight of the edge
    """
    return item[2]
