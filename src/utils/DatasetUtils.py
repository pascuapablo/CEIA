from enum import Enum
import numpy as np


class DatasetUtils:

    def __init__(self, dataset: np.ndarray):
        self.ds = dataset

    def get_dataset(self) -> np.ndarray:
        return self.ds

    def substract_mean(self, axis) -> np.ndarray:
        mean = np.mean(self.ds, axis, keepdims=True)
        return self.ds - mean

    def normilize_std(self, axis) -> np.ndarray:
        std = np.std(self.ds, axis, keepdims=True)
        self.ds = self.ds / std
        return self.ds

    def pca(self, n_dim):
        x2 = self.substract_mean(axis=0)
        cov_1 = np.cov(x2.T)
        w, v = np.linalg.eig(cov_1)
        idx = w.argsort()[::-1]
        v = v[:, idx]
        return np.matmul(x2, v[:, :n_dim])

    def k_means(self, clusters: int):

        centroid_indexes = np.random.uniform(0, self.ds.shape[0], clusters).astype(int)
        centroids = self.ds[centroid_indexes, :]
        expanded_centroids = centroids[:, None]
        gradient = np.zeros(11)
        median = 0
        for j in range(10):
            distance_matrix = np.sqrt(np.sum((expanded_centroids - self.ds) ** 2, axis=2))

            gradient[j] = np.median(distance_matrix) - median
            median = np.median(distance_matrix)

            arg_min = np.argmin(distance_matrix, axis=0)

            for i in range(clusters):
                centroids[i, :] = np.mean(self.ds[arg_min == i, :], axis=0)

        print(gradient)
        return centroids

