import unittest

from src.machineLearning.metrics.ErrorMean import ErrorMean
from src.machineLearning.metrics.ErrorMedian import ErrorMedian
from src.machineLearning.metrics.MSE import MSE
from src.machineLearning.regresion.LinearRegression import LinearRegression
from src.utils.DatasetUtils import DatasetUtils
import numpy as np
import matplotlib.pyplot as plot


class Ejercicios(unittest.TestCase):
    def test_something(self):
        ds = DatasetUtils(path="resources/dataset.csv")

        x_train, x_test = ds.split(80)
        print(x_train[:, 1].ndim, x_train[:, 1:2].ndim)
        lr = LinearRegression()
        lr.fit(x_train[:, 1], x_train[:, 2])
        y_predict = lr.predict(x_test[:, 1])
        y_real = x_test[:, 2]

        plot.scatter(x_test[:, 1], y_real, c="blue")
        plot.scatter(x_test[:, 1], y_predict, c="red")
        plot.show()

        metrics = [MSE(), ErrorMedian(), ErrorMean()]

        for metric in metrics:
            metric_name = type(metric).__name__
            print(metric_name, ":", metric(target=y_real, prediction=y_predict))


if __name__ == '__main__':
    unittest.main()
