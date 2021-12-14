import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_circles, make_moons, make_blobs

from PysparkMSTfordensegraphsfast import create_mst, create_clustering, mst_create
from boruvka import boruvka
from helpers import create_distance_matrix, plot_mst, plot_clustering

level_count = 5
noise_levels = [i * (0.1 / level_count) for i in range(level_count + 1)]


def make_ansi(n_samples, noise):
    X, y = make_blobs(n_samples=n_samples, random_state=170,
                      cluster_std=[noise * i for i in range(1, 4)])
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    return X_aniso, y


def generate_test_data(n_samples=2000, do_plot=True):
    datasets = []
    datasets = datasets + [(2, make_circles(n_samples=n_samples, factor=.5,
                                            noise=noise)) for noise in noise_levels]
    datasets = datasets + [(2, make_moons(n_samples=n_samples,
                                          noise=noise)) for noise in noise_levels]
    datasets = datasets + [(3, make_blobs(n_samples=n_samples, random_state=5,
                                          cluster_std=[noise * i for i in range(1, 4)])) for noise in noise_levels]
    datasets = datasets + [(3, make_ansi(n_samples, noise)) for noise in noise_levels]

    if do_plot:
        plot_datasets(datasets)

    return datasets


def plot_datasets(p_datasets):
    height = int(len(p_datasets) / (level_count + 1))
    width = level_count + 1
    plt.figure(figsize=(5* width + 2, height * 5 + 2))
    plt.subplots_adjust(
        left=0.02, right=0.98, bottom=0.05, top=0.96, wspace=0.05, hspace=0.1
    )
    for i, dataset in enumerate(p_datasets):
        dataset = dataset[1]
        plt.subplot(height, width, i + 1)
        plt.scatter(x=dataset[0][:, 0], y=dataset[0][:, 1], s=2)
        plt.xticks(())
        plt.yticks(())
        if i <= level_count:
            plt.title(f"Noise amount: {round(noise_levels[i], 2)}", size=24)
    plt.show()


def plot_clusters(vertices, clustering):
    height = int(len(clustering) / (level_count + 1))
    width = level_count + 1
    print(height, width)
    plt.figure(figsize=(5 * width + 2, height * 5 + 2))
    plt.subplots_adjust(
        left=0.02, right=0.98, bottom=0.05, top=0.96, wspace=0.05, hspace=0.1
    )
    for i, dataset in enumerate(clustering):
        plt.subplot(height, width, i + 1)
        x = []
        y = []
        c = []
        area = []
        colors = ["dodgerblue", "r", "g", "c", "m", "y", "k", "darkorange", "dodgerblue", "deeppink", "khaki", "purple",
                  "springgreen", "tomato", "slategray"]
        for j in range(len(dataset)):
            cluster = dataset[j]
            for v in cluster:
                x.append(vertices[i][0][v][0])
                y.append(vertices[i][0][v][1])
                area.append(0.1)
                c.append(colors[j % len(colors)])

        plt.scatter(x, y, c=c, s=2)
        plt.xticks(())
        plt.yticks(())
        if i <= level_count:
            plt.title(f"Noise amount: {round(noise_levels[i], 2)}", size=18)
    plt.show()


def get_stats(dataset, clusters):
    true_clusters = dataset[1][1]
    for i, cluster in enumerate(true_clusters):
        true_clusters[i] -= 1
    true_cluster_size = [0]*dataset[0]
    for cluster in range(dataset[0]):
        true_cluster_size[cluster] = np.sum(true_clusters == cluster)
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for cluster in clusters:
        classification = [0]*dataset[0]
        for vertex in cluster:
            vertex_class = true_clusters[vertex]
            classification[vertex_class] += 1
        tp = max(classification)
        fp = sum(classification) - tp
        fn = true_cluster_size[classification.index(tp)] - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn
    precision = total_tp / (total_tp + total_fp)
    recall = total_tp / (total_tp + total_fn)
    print("print precision : ", precision)
    print("print recall : ", recall)


def clustering_mst_mrc(dataset):
    mst = boruvka(dataset[1][0])
    # mst = mst_create(dataset[1][0])
    k = dataset[0]
    clusters = create_clustering(len(dataset[1][0]), mst, k)
    get_stats(dataset, clusters)
    return clusters


def bench():
    datasets = generate_test_data(do_plot=True)
    results = []
    for i, dataset in enumerate(datasets):
        results.append(clustering_mst_mrc(dataset))
    res = []
    for dataset in datasets:
        res.append(dataset[1])
    plot_clusters(res, results)


def bench_circle_probability(n_samples, noise, rounds):
    datasets = []
    datasets = datasets + [(2, make_circles(n_samples=n_samples, factor=.5,
                                            noise=noise)) for _ in range(rounds)]
    count_failed = 0
    res = []
    results = []
    for dataset in datasets:
        res.append(dataset[1])

    for i, dataset in enumerate(datasets):
        clusters = clustering_mst_mrc(dataset)
        results.append(clusters)
        for cluster in clusters:
            if len(cluster) == 1:
                count_failed += 1
        print(f"CLUSTERING FAILED {count_failed} / {i + 1}")
    plot_clusters(res, results)


def load_txt_to_data(filepath):
    dataset = []
    with open(filepath) as file:
        for line in file:
            line = line.strip("\t")
            line = line.strip("\n")
            line = line.strip(" ")
            line = line.split("   ")
            line = [float(line[0]), float(line[1])]
            dataset.append(line)
    clustering = []
    with open("a3_part.txt") as file:
        for line in file:
            clustering.append(int(line))
    dataset = np.array(dataset)
    plt.scatter(x=dataset[:, 0], y=dataset[:, 1], s=1)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    return 50, [dataset, np.array(clustering)]


def custom_dataset():
    dataset = load_txt_to_data("a3.txt")
    clusters = clustering_mst_mrc(dataset)
    get_stats(dataset, clusters)
    x = []
    y = []
    c = []
    area = []
    colors = ["dodgerblue", "r", "g", "c", "m", "y", "k", "darkorange", "dodgerblue", "deeppink", "khaki", "purple",
              "springgreen", "tomato", "slategray"]
    for j in range(len(clusters)):
        cluster = clusters[j]
        for v in cluster:
            x.append(dataset[1][0][v][0])
            y.append(dataset[1][0][v][1])
            area.append(0.1)
            c.append(colors[j % len(colors)])

    plt.scatter(x, y, c=c, s=1)
    plt.xticks(())
    plt.yticks(())
    plt.show()


if __name__ == '__main__':
    custom_dataset()
    # bench_circle_probability(1500, 0.07, 96)
    # bench()
