import unittest
import numpy as np
import src.utils.ClusterUtils as C
from src.utils.FileUtils import FileUtils


class Ejercicios(unittest.TestCase):
    def setUp(self):
        self.SAMPLES = 100000
        self.CLUSTER_DIM = 4

    def test_ejercicio1(self):
        dataset = C.ClusterUtils().build_synthetic_cluster(self.CLUSTER_DIM, self.SAMPLES)
        a = dataset.shape == np.array([self.CLUSTER_DIM, self.CLUSTER_DIM, self.SAMPLES])

        print(dataset[:, :, 10])
        # [
        #   [10.55668902 - 1.5955966 - 0.5000201 - 0.22836082]
        #   [1.50070782  9.75542079  0.21029596  0.43774309]
        #   [-0.89518945 - 0.45219363  10.77350883 - 0.10070804]
        #   [-0.50700621  1.27458278 - 1.04033711 10.24162002]
        # ]
        self.assertTrue(a.all())

    def test_ejercicio2y3(self):
        x = C.ClusterUtils().build_synthetic_cluster(self.CLUSTER_DIM, self.SAMPLES)
        uniform = np.random.uniform(0, 1, (self.CLUSTER_DIM, self.CLUSTER_DIM, self.SAMPLES))
        uniform = np.where(uniform < 0.1, np.NaN, 0)
        x = x + uniform

        FileUtils.save_dataset(x, "./resoruces/data.pkl")

    def test_ejercicios4a7(self):
        x = FileUtils.load_dataset("./resoruces/data.pkl")
        x_average = np.nanmean(x, axis=0)
        x_average = x_average[None, :]
        x_average = x_average.repeat(4, 0)
        x[np.isnan(x)] = x_average[np.isnan(x)]

        x_l2_norm = np.linalg.norm(x, axis=0)
        x_mean = x.mean(axis=0)
        x_std = x.std(axis=0)


if __name__ == '__main__':
    unittest.main()
