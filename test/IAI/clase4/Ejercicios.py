import unittest
import numpy as np
import matplotlib.pyplot as plot

from src.machineLearning.metrics.MSE import MSE
from src.machineLearning.regresion.LinearRegression import LinearRegression
from src.machineLearning.regresion.LinearRegressionAfine import LinearRegressionAffine
from src.utils.DatasetUtils import DatasetUtils


class MyTestCase(unittest.TestCase):

    def test_something(self):
        N = 6
        lr_prediction_error = []
        lra_prediction_error = []
        ds = DatasetUtils(path="../clase3/resources/dataset.csv")

        ds_train, ds_test = ds.split(80)

        for j in range(1, N):
            x_train = ds_train[:, 1:2]
            x_test = ds_test[:, 1:2]
            for i in range(1, j):
                x_train = np.append(x_train, ds_train[:, 1:2] ** (i + 1), axis=1)
                x_test = np.append(x_test, ds_test[:, 1:2] ** (i + 1), axis=1)

            y_test = ds_test[:, 2:3]
            y_train = ds_train[:, 2:3]
            lr = LinearRegression()
            lr.fit(x_train, y_train)
            y_lr_predict = lr.predict(x_test)
            lr_prediction_error.append(MSE()(y_test, y_lr_predict))

            lra = LinearRegressionAffine()
            lra.fit(x_train, y_train)
            y_lra_predict = lra.predict(x_test)
            lra_prediction_error.append(MSE()(y_test, y_lra_predict))


            # plot.figure(j)
            # plot.scatter(x_test[:, 1], y_test, c="blue")
            # plot.scatter(x_test[:, 1], y_lr_predict, c="red")
            # plot.show()

        plot.figure("MSE")
        plot.plot(range(1, N), lr_prediction_error)
        plot.plot(range(1, N), lra_prediction_error)
        plot.xlabel("Cantidad de dimensiones")
        plot.ylabel("MSE")
        plot.show()




if __name__ == '__main__':
    unittest.main()
