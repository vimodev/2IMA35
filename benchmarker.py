from matplotlib import pyplot as plt
from sklearn.datasets import make_circles, make_moons, make_blobs

from PysparkMSTfordensegraphsfast import create_mst, create_clustering
from boruvka import boruvka
from helpers import create_distance_matrix, plot_mst, plot_clustering

level_count = 5
noise_levels = [i * (0.2 / level_count) for i in range(level_count + 1)]


def generate_test_data(n_samples=2000, do_plot=True):
    datasets = []
    datasets = datasets + [make_circles(n_samples=n_samples, factor=.5,
                                        noise=noise) for noise in noise_levels]
    datasets = datasets + [make_moons(n_samples=n_samples,
                                      noise=noise) for noise in noise_levels]
    datasets = datasets + [make_blobs(n_samples=n_samples, random_state=5,
                                      cluster_std=[noise * i for i in range(1, 4)]) for noise in noise_levels]

    if do_plot:
        plot_datasets(datasets)

    return datasets


def plot_datasets(p_datasets):
    height = int(len(p_datasets) / (level_count + 1))
    width = level_count + 1
    plt.figure(figsize=(3.5 * width + 2, height * 5 + 2))
    plt.subplots_adjust(
        left=0.02, right=0.98, bottom=0.05, top=0.96, wspace=0.05, hspace=0.1
    )
    for i, dataset in enumerate(p_datasets):
        plt.subplot(height, width, i + 1)
        plt.scatter(x=dataset[0][:, 0], y=dataset[0][:, 1])
        plt.xticks(())
        plt.yticks(())
        if i <= level_count:
            plt.title(f"Noise amount: {round(noise_levels[i], 2)}", size=18)
    plt.show()


def plot_clusters(vertices, clustering):
    height = int(len(clustering) / (level_count + 1))
    width = level_count + 1
    plt.figure(figsize=(3.5 * width + 2, height * 5 + 2))
    plt.subplots_adjust(
        left=0.02, right=0.98, bottom=0.05, top=0.96, wspace=0.05, hspace=0.1
    )
    for i, dataset in enumerate(clustering):
        plt.subplot(height, width, i + 1)
        x = []
        y = []
        c = []
        area = []
        colors = ["g", "b", "r", "c", "m", "y", "k", "darkorange", "dodgerblue", "deeppink", "khaki", "purple",
                  "springgreen", "tomato", "slategray"]
        for j in range(len(dataset)):
            cluster = dataset[j]
            for v in cluster:
                x.append(vertices[i][0][v][0])
                y.append(vertices[i][0][v][1])
                area.append(0.1)
                c.append(colors[j % len(colors)])

        plt.scatter(x, y, c=c)
        plt.xticks(())
        plt.yticks(())
        if i <= level_count:
            plt.title(f"Noise amount: {round(noise_levels[i], 2)}", size=18)
    plt.show()


def clustering_mst_mrc(dataset):
    mst = boruvka(dataset[0])
    k = 2
    clusters = create_clustering(len(dataset[0]), mst, k)
    return clusters


def bench():
    datasets = generate_test_data(do_plot=True)
    results = []
    for i, dataset in enumerate(datasets):
        results.append(clustering_mst_mrc(dataset))
        if i % 5 == 0 and i != 0:
            plot_clusters(datasets, results)


if __name__ == '__main__':
    bench()
