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
    no_structure = np.random.rand(n_samples, 2), None

    # Anisotropicly distributed data
    random_state = 170
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = make_blobs(n_samples=n_samples,
                        cluster_std=[1.0, 2.5, 0.5],
                        random_state=random_state)

    plt.figure(figsize=(9 * 2 + 3, 13))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.95, wspace=.05,
                        hspace=.01)

    datasets = [
        (noisy_circles, {'damping': .77, 'preference': -240,
                         'quantile': .2, 'n_clusters': 2,
                         'min_samples': 20, 'xi': 0.25}),
        (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
        (varied, {'eps': .18, 'n_neighbors': 2,
                  'min_samples': 5, 'xi': 0.035, 'min_cluster_size': .2}),
        (aniso, {'eps': .15, 'n_neighbors': 2,
                 'min_samples': 20, 'xi': 0.1, 'min_cluster_size': .2}),
        (blobs, {}),
        (no_structure, {})]

    return datasets


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

def plot_clustering(vertices, clustering):
    x = []
    y = []
    c = []
    area = []
    colors = ["g", "b", "r", "c", "m", "y", "k", "darkorange", "dodgerblue", "deeppink", "khaki", "purple",
              "springgreen", "tomato", "slategray"]
    for i in range(len(clustering)):
        cluster = clustering[i]
        for v in cluster:
            x.append(float(vertices[v][0]))
            y.append(float(vertices[v][1]))
            area.append(0.1)
            c.append(colors[i % len(colors)])
    plt.scatter(x, y, c=c)
    plt.show()
    return

def plot_mst(vertices, mst, intermediate=False, plot_cluster=False):
    x = []
    y = []
    c = []
    area = []
    colors = ["g", "b", "r", "c", "m", "y", "k", "darkorange", "dodgerblue", "deeppink", "khaki", "purple",
              "springgreen", "tomato", "slategray"]
    for i in range(len(vertices)):
        x.append(float(vertices[i][0]))
        y.append(float(vertices[i][1]))
        area.append(0.1)
        c.append("black")
    plt.scatter(x, y, c=c, s=area)
    if intermediate:
        cnt = 0
        for m in mst:
            for i in range(len(m)):
                linex = [float(x[int(m[i][0])])]
                liney = [float(y[int(m[i][0])])]
                linex.append(float(x[int(m[i][1])]))
                liney.append(float(y[int(m[i][1])]))
                plt.plot(linex, liney, colors[cnt])
            cnt = (cnt + 1) % len(colors)
    else:
        # TODO
        if plot_cluster:
            edges = sorted(mst, key=get_key, reverse=True)
            total_length = 0
            for edge in edges:
                total_length += edge[2]
            average_length = total_length / len(edges)
            print(average_length)
            print(edges[0], edges[1], edges[2], edges[3])
            print("Not yet implemented")
        else:
            for i in range(len(mst)):
                linex = [float(x[int(mst[i][0])])]
                liney = [float(y[int(mst[i][0])])]
                linex.append(float(x[int(mst[i][1])]))
                liney.append(float(y[int(mst[i][1])]))
                plt.plot(linex, liney)
    plt.show()
    return


def get_key(item):
    """
    returns the sorting criteria for the edges. All edges are sorted from small to large values
    :param item: one item
    :return: returns the weight of the edge
    """
    return item[2]
