import unittest

from src.machineLearning.metrics.MeanError import MeanError
from src.machineLearning.metrics.MedianError import MedianError
from src.machineLearning.metrics.MSE import MSE
from src.machineLearning.regresion.LinearRegression import LinearRegression
from src.machineLearning.regresion.LinearRegressionAfine import LinearRegressionAffine
from src.utils.DatasetUtils import DatasetUtils
import numpy as np
import matplotlib.pyplot as plot


class Ejercicios(unittest.TestCase):
    def setUp(self):
        np.set_printoptions(precision=2)
        np.set_printoptions(suppress=True)

    def test_something(self):
        ds = DatasetUtils(path="resources/dataset.csv")

        x_train, x_test = ds.split(80)
        y_real = x_test[:, 2]

        lr = LinearRegression()
        lr.fit(x_train[:, 1], x_train[:, 2])
        y_lr_predict = lr.predict(x_test[:, 1])

        lra = LinearRegressionAffine()
        lra.fit(x_train[:, 1], x_train[:, 2])
        y_lra_predict = lra.predict(x_test[:, 1])

        plot.scatter(x_test[:, 1], y_real, c="blue")
        plot.scatter(x_test[:, 1], y_lr_predict, c="red")
        plot.scatter(x_test[:, 1], y_lra_predict, c="yellow")
        plot.show()

        metrics = [MSE(), MedianError(), MeanError()]

        print("Metric", "LinearRegression", "LinearRegressionAffine")
        for metric in metrics:
            print(metric.get_name(), metric(target=y_real, prediction=y_lr_predict),
                  metric(target=y_real, prediction=y_lra_predict))

        # Metric        LinearRegression        LinearRegressionAffine
        # MSE           3.64650918978914        3.4685120979301836
        # MedianError   0.007340468339615436    -0.01885682436420999
        # MeanError     0.021328249515553076    -0.0025382219110967837
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
