import numpy as np
from src.machineLearning.IMLBaseModel import IMLBaseModel


class Kmeans(IMLBaseModel):

    def __init__(self, n_clusters):
        super().__init__()
        self.n_clusters = n_clusters

    def fit(self, x, y=None) -> np.ndarray:
        centroid_indexes = np.random.uniform(0, x.shape[0], self.n_clusters).astype(int)
        centroids = x[centroid_indexes, :]
        expanded_centroids = centroids[:, None]
        gradient = np.zeros(11)
        median = 0

        for j in range(10):
            distance_matrix = np.sqrt(np.sum((expanded_centroids - x) ** 2, axis=2))

            gradient[j] = np.median(distance_matrix) - median
            median = np.median(distance_matrix)

            arg_min = np.argmin(distance_matrix, axis=0)

            for i in range(self.n_clusters):
                centroids[i, :] = np.mean(x[arg_min == i, :], axis=0)

        print(gradient)
        self.model = centroids
        return centroids

    def predict(self, x):
        raise NotImplementedError
