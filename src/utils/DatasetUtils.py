import numpy as np


class DatasetUtils:

    def __init__(self, dataset):
        self.ds = dataset

    def get_dataset(self) -> np.ndarray:
        return self.ds

    def substract_mean(self, axis) -> np.ndarray:
        mean = np.mean(self.ds, axis, keepdims=True)
        self.ds = self.ds - mean
        return self.ds

    def normilize_std(self, axis) -> np.ndarray:
        std = np.std(self.ds, axis, keepdims=True)
        self.ds = self.ds / std
        return self.ds



    #
    # MAX_ITERATIONS = 10
    #
    # def k_means(X, n_clusters):
    #     centroids = np.eye(n_clusters, X.shape[1])
    #     print(centroids)
    #     for i in range(MAX_ITERATIONS):
    #         print("Iteration # {}".format(i))
    #         centroids, cluster_ids = k_means_loop(X, centroids)
    #         print(centroids)
    #     return centroids, cluster_ids
    #
    # def k_means_loop(X, centroids):
    #     # find labels for rows in X based in centroids values
    #     expanded_centroids = centroids[:, None]
    #     distances = np.sqrt(np.sum((expanded_centroids - X) ** 2, axis=2))
    #     arg_min = np.argmin(distances, axis=0)
    #     # recompute centroids
    #     for i in range(centroids.shape[0]):
    #         centroids[i] = np.mean(X[arg_min == i, :], axis=0)
    #     return centroids, arg_min
    #
    # A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # C = np.array([[1, 0, 2], [1, 3, 4]])
    #
    # # print(A)
    # # print(C[:,None])
    #
    # print(np.sqrt(np.sum((C[:, None] - A) ** 2, axis=2)))
    #
    # A = np.array([[1, 0, 0, 0], [0, 0, 0, 1]])
    # B = A * 10
    # C = np.repeat(B, 10, axis=0)
    #
    # R = np.random.normal(0, 1, C.shape)
    #
    # print(C.shape)
    # print(C + R)
    #
    # print(k_means(C + R, 2))
    #
    # # %%
    # def exp(lambda_val, samples):
    #     U = np.random.uniform(0, samples)
    #     return -np.log2(1 - U) / lambda_val
    #
    # plot(exp(2, 100))
    #
    # # %%
    # def normalize(x):
    #     mean = np.mean(x, axis=0)
    #     std = np.std(x, axis=0)
    #
    #     return (x - mean) / std
    #
    # def pca(X):
    #     a = X
    #     a = a - a.mean(axis=0)
    #     w, v = np.linalg.eig(np.cov(a.T));
    #     idx = w.argsort()[::-1]
    #     print(idx)
    #     w = w[idx]
    #     v = v[:, idx]
    #
    #     print(np.matmul(a, v[:, :2]))
    #     return 1
    #
    # pca(np.array([[1, 0, 1], [0, 1, 1]]))
    #
