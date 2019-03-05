import matplotlib.pyplot as plt
import numpy as np


class Plotting:

    def __init__(self, data_scaled, new_clusters, single_cluster=False):

        self.data_scaled = data_scaled

        centers, labels = self.get_kmeans()

        if not single_cluster:
            for cluster in new_clusters:
                self.schema(data_scaled, centers, labels, cluster)
        else:
            self.schema(data_scaled, centers, labels, new_clusters)

    @staticmethod
    def schema(data_scaled, centers, labels, clusters):
        plt.title('Score: {}'.format(clusters[-1]))
        plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels)
        plt.scatter(np.array(centers)[:, 0], np.array(centers)[:, 1], c=np.unique(labels), s=200, alpha=0.5)

        for el in clusters[1]:
            plt.scatter(el[0], el[1], marker='X', s=200)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    def get_kmeans(self):
        from sklearn.cluster import KMeans
        n_clusters = 3
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(self.data_scaled)
        return kmeans.cluster_centers_, kmeans.labels_