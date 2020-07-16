import unittest
import numpy as np
import src.machineLearning.clustering.ClusterUtils as C
from src.machineLearning.clustering.Kmeans import Kmeans
from src.utils.DatasetUtils import DatasetUtils
from src.utils.RandomUtils import RandomUtils
import matplotlib.pyplot as plot


class Ejercicios(unittest.TestCase):
    def setUp(self):
        self.SAMPLES = 100000
        self.CLUSTER_DIM = 4
        self.CLUSTER_N_CENTROIDS = 3
        self.SAVE_PATH = "./resoruces/data.pkl"

        # np.set_printoptions(precision=2)
        # np.set_printoptions(suppress=True)

    def test_ejercicio1(self):
        dataset = C.ClusterUtils().build_synthetic_cluster(self.CLUSTER_DIM, self.CLUSTER_N_CENTROIDS, self.SAMPLES)
        a = dataset.shape == np.array([self.SAMPLES, self.CLUSTER_DIM])
        print(dataset.shape)
        self.assertTrue(a.all())

    def test_ejercicio2y3(self):
        x = C.ClusterUtils().build_synthetic_cluster(self.CLUSTER_DIM, self.CLUSTER_N_CENTROIDS, self.SAMPLES)
        uniform = np.random.uniform(0, 1, (self.SAMPLES, self.CLUSTER_DIM))
        uniform = np.where(uniform < 0.1, np.NaN, 0)
        x = x + uniform
        print(x)
        DatasetUtils.save_dataset(x, self.SAVE_PATH)

    def test_ejercicios4a7(self):
        x = DatasetUtils.load_dataset(self.SAVE_PATH)
        x_average = np.nanmean(x, axis=0)

        x = np.where(np.isnan(x), x_average, x)

        x_l2_norm = np.linalg.norm(x, axis=0)
        x_mean = x.mean(axis=0)
        x_std = x.std(axis=0)

    def test_ejercicio7y8(self):
        x = DatasetUtils.load_dataset(self.SAVE_PATH)
        exp = RandomUtils.exp(0.1, self.SAMPLES).reshape(self.SAMPLES, 1)
        exp = np.append(x, exp, axis=1)
        plot.hist(exp[:, -1], 50)
        plot.show()

    def test_ejercicio9(self):
        x = C.ClusterUtils().build_synthetic_cluster(self.CLUSTER_DIM,
                                                     self.CLUSTER_N_CENTROIDS,
                                                     self.SAMPLES,
                                                     centroids_distance=50)

        exp = RandomUtils.exp(500, self.SAMPLES).reshape(self.SAMPLES, 1)
        x = np.append(x, exp, axis=1)

        x_pca = DatasetUtils(x).pca(2)

        plot.show()
        plot.scatter(x_pca[:, 0], x_pca[:, 1])

    def test_ejercicio10(self):
        x = C.ClusterUtils().build_synthetic_cluster(2, 2, self.SAMPLES, centroids_distance=10, noise_level=1)

        plot.scatter(x[:, 0], x[:, 1])

        centroids = Kmeans(2).fit(x)
        plot.scatter(centroids[:, 0], centroids[:, 1], c='red')

        plot.show()


if __name__ == '__main__':
    unittest.main()
