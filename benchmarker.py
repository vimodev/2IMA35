import random
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_circles, make_moons, make_blobs

from PysparkMSTfordensegraphsfast import create_mst, create_clustering, mst_create
from boruvka import boruvka
from helpers import create_distance_matrix, plot_mst, plot_clustering

level_count = 5
noise_levels = [0.0 + i * (1 / level_count) for i in range(level_count + 1)]


def make_ansi(n_samples, noise):
    X, y = make_blobs(n_samples=n_samples, random_state=random.randint(1,100),
                      cluster_std=[noise * i for i in range(1, 4)])
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    return X_aniso, y


def generate_test_data(n_samples=2000, do_plot=True):
    datasets = []
    # datasets = datasets + [(2, make_circles(n_samples=n_samples, factor=.5,
                                            # noise=noise)) for noise in noise_levels]
    # datasets = datasets + [(2, make_moons(n_samples=n_samples,
                                          # noise=noise)) for noise in noise_levels]
    # datasets = datasets + [(3, make_blobs(n_samples=n_samples, random_state=random.randint(1, 100),
    #                                       cluster_std=[noise * i for i in range(1, 4)])) for noise in noise_levels]
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
            plt.title(f"Noise amount: {round(noise_levels[i], 4)}", size=24)
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
            plt.title(f"Noise amount: {round(noise_levels[i], 5)}", size=24)
    plt.show()


def get_stats(dataset, clusters):
    true_clusters = dataset[1][1]
    tp = 0
    fp = 0
    fn = 0
    for u in range(len(true_clusters)):
        for v in range(len(true_clusters)):
            if (u==v):
                continue
            inferred = False
            for c in range(len(clusters)):
                if u in clusters[c]:
                    inferred = (v in clusters[c])
                    break
            truth = (true_clusters[u] == true_clusters[v])
            if (truth and inferred):
                tp += 1
            elif (truth):
                fn += 1
            elif (inferred):
                fp += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print("PRECISION: " + str(precision))
    print("RECALL: " + str(recall))

    return precision, recall


def clustering_mst_mrc(dataset):
    mst = boruvka(dataset[1][0])
    # mst = mst_create(dataset[1][0])
    k = dataset[0]
    clusters = create_clustering(len(dataset[1][0]), mst, k)
    #get_stats(dataset, clusters)
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
    datasets = datasets + [(3, make_blobs(n_samples=n_samples, random_state=random.randint(1, 200),
                                          cluster_std=[noise * i for i in range(1, 4)])) for _ in range(rounds)]
    # datasets = datasets + [(2, make_circles(n_samples=n_samples, factor=.5,
    #                                         noise=noise)) for _ in range(rounds)]
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

def bench_noise():
    runs_per_level = 100
    level_count = 100
    levels = [i * (1 / level_count) for i in range(level_count + 1)]
    precisions = [0 for i in range(len(levels))]
    recalls = [0 for i in range(len(levels))]
    for i in range(len(levels)):
        for j in range(runs_per_level):
            print("Start creating MST...")
            timestamp = datetime.now()
            print("level " + str(levels[i]) + " - run " + str(j))
            # dataset = (2, make_moons(n_samples=500,
            #                               noise=levels[i]))
            # dataset = (3, make_blobs(n_samples=250, random_state=random.randint(1,100),
                       # cluster_std=[levels[i] * _ for _ in range(1, 4)]))
            # dataset = (3, make_ansi(n_samples=250, noise=[levels[i]]))
            dataset = (3, make_ansi(n_samples=250, noise=levels[i]))
            clusters = clustering_mst_mrc(dataset)
            precision, recall = get_stats(dataset, clusters)
            precisions[i] += precision
            recalls[i] += recall
            print("Found MST in: ", datetime.now() - timestamp)
        precisions[i] = precisions[i] / runs_per_level
        recalls[i] = recalls[i] / runs_per_level
    plt.plot(levels, precisions, label="Precision")
    plt.plot(levels, recalls, label="Recall")
    plt.legend()
    plt.show()

used_rounds = []

def bench_rounds():
    for i in range(1,100):
        samples = i * 50
        dataset = make_moons(n_samples=samples, noise=0.2)
        mst_create(dataset[0])


if __name__ == '__main__':
    # custom_dataset()
    # bench_circle_probability(2000, 0.075, 6)
    # bench()
    bench_rounds()
    #generate_test_data(do_plot=True)
