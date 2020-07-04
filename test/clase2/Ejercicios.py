import unittest
import numpy as np
import src.utils.ClusterUtils as C


class Ejercicios(unittest.TestCase):

    def test_ejercicio1(self):
        dataset = C.ClusterUtils().build_synthetic_cluster(4, 100000)
        a = dataset.shape == np.array([4, 4, 100000])

        print(dataset[:, :, 10])
        # [
        #   [10.55668902 - 1.5955966 - 0.5000201 - 0.22836082]
        #   [1.50070782  9.75542079  0.21029596  0.43774309]
        #   [-0.89518945 - 0.45219363  10.77350883 - 0.10070804]
        #   [-0.50700621  1.27458278 - 1.04033711 10.24162002]
        # ]
        self.assertTrue(a.all())


if __name__ == '__main__':
    unittest.main()
