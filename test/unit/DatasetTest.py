import unittest
import numpy as np
import src.utils.DatasetUtils as D


class DatasetTest(unittest.TestCase):

    def test_substract_mean(self):
        x = np.array([[1, 2, 3], [1, 2, 3]])
        ds = D.DatasetUtils(x)
        ds.substract_mean(axis=0)

        self.assertEqual(np.count_nonzero(ds.get_dataset().mean(axis=0)), 0)

    def test_substract_mean2(self):
        x = np.array([[1, 2, 3], [1, 2, 3]])
        ds = D.DatasetUtils(x)
        ds.substract_mean(axis=1)
        self.assertEqual(np.count_nonzero(ds.get_dataset().mean(axis=1)), 0)

    def test_normalize(self):
        x = np.array([[1, 2, 3], [1.1, 2.1, 3.1]])
        ds = D.DatasetUtils(x)
        x_normilized = ds.normilize_std(axis=0)
        std = x_normilized.std(axis=0)
        self.assertEqual(std.sum(), np.max(std.shape))

    def test_normalize2(self):
        x = np.array([[1, 2, 3], [1.1, 2.1, 3.1]])
        ds = D.DatasetUtils(x)
        x_normilized = ds.normilize_std(axis=1)

        std = x_normilized.std(axis=1)
        self.assertEqual(std.sum(), np.max(std.shape))


if __name__ == '__main__':
    unittest.main()
