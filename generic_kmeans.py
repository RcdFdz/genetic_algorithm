import numpy as np
import random


class GenericKmeans:

    def __init__(self, points, num_centroids, w, r):
        self.points = points
        self.num_centroids = num_centroids
        self.wi = w
        self.ri = r
        self.centroids = self.initialize_centroids()
        self.distances = []

    def initialize_centroids(self):
        """returns k centroids from the initial points"""
        return random.choices(self.points, k=self.num_centroids)

    def get_distance(self):
        """returns the points distance"""
        return [np.sum(np.multiply(self.wi, np.power(abs(np.subtract(self.centroids, v)), self.ri)), axis=1)
                for i, v in enumerate(np.matrix(self.points))]

    def closest_centroid(self):
        """returns an array containing the index to the nearest centroid for each point"""
        self.distances = self.get_distance()
        return np.argmin(self.distances, axis=1)

    def new_centroids(self, precision=4):
        points_closet_centroid = np.hstack([self.points, self.closest_centroid()])
        points_by_cluster = [points_closet_centroid[np.where(points_closet_centroid[:, -1] == idx), :-1]
                             for idx in np.unique(points_closet_centroid[:, -1])]

        new_centroids = [np.average(cluster[0], axis=0) for cluster in points_by_cluster]

        for new_centroid in new_centroids:
            if np.round(new_centroid, precision) not in np.round(self.centroids, precision):
                self.centroids = new_centroids
                self.new_centroids()
        return new_centroids
