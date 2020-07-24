import unittest
import numpy as np
import matplotlib.pyplot as plot

from src.machineLearning.metrics.MSE import MSE
from src.machineLearning.regresion.LinearRegression import LinearRegression
from src.utils.DatasetUtils import DatasetUtils


class MyTestCase(unittest.TestCase):

    def test_something(self):
        N = 10
        prediction_error = []
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
            print("x train", x_train.shape)
            lr = LinearRegression()
            lr.fit(x_train, y_train)
            print("W", lr.model.shape)
            y_lr_predict = lr.predict(x_test)
            prediction_error.append(MSE()(y_test, y_lr_predict))

            # plot.figure(j)
            # plot.scatter(x_test[:, 1], y_test, c="blue")
            # plot.scatter(x_test[:, 1], y_lr_predict, c="red")
            # plot.show()

        plot.figure("MSE")
        plot.plot(range(1, N), prediction_error)
        plot.xlabel("Cantidad de dimensiones")
        plot.ylabel("MSE")
        plot.show()



if __name__ == '__main__':
    unittest.main()
