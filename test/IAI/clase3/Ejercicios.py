import unittest

from src.utils.DatasetUtils import DatasetUtils
from src.utils.FileUtils import FileUtils


class Ejercicios(unittest.TestCase):
    def test_something(self):
        x = FileUtils.load_fromCSV("resources/dataset.csv")
        ds = DatasetUtils(x[1:, 1:])



        print(x)

        # W = np.matmul(np.invert(np.matmul(x.T, x)), x.T)


if __name__ == '__main__':
    unittest.main()
