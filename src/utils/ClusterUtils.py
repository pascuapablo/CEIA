import numpy as np


class ClusterUtils:

    def build_synthetic_cluster(self, ndim, ncentroids, nsamples, centroids_distance=10, noise_level=1) -> np.ndarray:
        x = np.eye(ndim)[:ncentroids, :]
        x = x * centroids_distance
        x = np.repeat(x, np.ceil(nsamples / ncentroids), axis=0)
        x = x + np.random.normal(0, 1, (x.shape[0], ndim)) * noise_level
        return x[:nsamples, :]
