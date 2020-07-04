import numpy as np


class ClusterUtils:

    def build_synthetic_cluster(self, ndim, nsamples) -> np.ndarray:
        x = np.eye(ndim)
        x = x * 10
        x = x[..., np.newaxis]

        return x + np.random.normal(0, 1, (4, 4, nsamples))
