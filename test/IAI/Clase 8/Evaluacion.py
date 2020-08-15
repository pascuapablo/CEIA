import unittest

import matplotlib.pyplot as plot
import numpy as np

from src.machineLearning.algorithms.LeastSquares import LeastSquares
from src.machineLearning.algorithms.MiniBatchGradientDescent import MiniBatchGradientDescent
from src.machineLearning.metrics.MSE import MSE
from src.machineLearning.regresion.LinearRegressionAfine import LinearRegressionAffine
from src.utils.DatasetUtils import DatasetUtils


class MyTestCase(unittest.TestCase):
    def test_ejercicio3(self):
        ds = DatasetUtils(path="clase_8_dataset.csv")
        ds_train, ds_test = ds.split(80)

        x_train = ds_train[:, 0:1]
        y_train = ds_train[:, 1:2]

        x_test = ds_test[:, 0:1]
        y_test = ds_test[:, 1:2]

        plot.figure()
        minError = 1000000
        minErrorIndex = -1

        # El problema de regresion lineal se resuelve analiticamente a partir de tratar de minimizar
        # el error cuadratico medio, con lo cual es logico luego testear los modelos con esa medida del error
        error = MSE()
        models = []
        for i in range(1, 5):
            lr = LinearRegressionAffine(order=i, algorithm=LeastSquares())
            validation_mean_error, validation_min_error, lr = DatasetUtils.k_folds(x=x_train, y=y_train, k=5,
                                                                                   ml_object=lr,
                                                                                   error=error)

            models.append(lr)
            plot.scatter(i, validation_min_error, c='red', label='Modelo con menor error')
            plot.scatter(i, validation_mean_error, c='blue', label='Media del error')

            y_predict = lr.predict(x_test)

            plot.scatter(i, error(target=y_test, prediction=y_predict), c='green', label='Error de prediccion')

            if minError > validation_min_error:
                minError = validation_min_error
                minErrorIndex = i

        print("[test_something] El orden del modelo con menor error fue el ", minErrorIndex)
        # [test_something] El orden del modelo con menor error fue el  3
        bestModel = models[3]
        plot.figure()
        plot.scatter(x_test, y_test, c='red', label="target")
        plot.scatter(x_test, bestModel.predict(y_test), c='blue', label="prediccion")
        x = np.linspace(-400, 400, 1000)

        model = bestModel.model[0] * (x ** 3) + bestModel.model[1] * (x ** 2) + bestModel.model[2] * x + \
                bestModel.model[3]
        # plot.plot(model, c='green', label="modelo")
        print("[test_ejercicio3] modelo", bestModel.model)
        plot.legend()
        plot.show()

    def test_ejercicio4(self):
        ds = DatasetUtils(path="clase_8_dataset.csv")
        ds_train, ds_test = ds.split(80)

        x_train = ds_train[:, 0:1]
        y_train = ds_train[:, 1:2]

        x_test = ds_test[:, 0:1]
        y_test = ds_test[:, 1:2]

        lr = LinearRegressionAffine(order=1, algorithm=MiniBatchGradientDescent(learning_rate=0.0001, n_epochs=60000,
                                                                                n_batches=16))

        lr.fit(x_train, y_train)

        plot.figure()
        print("[test_ejercicio4] error", lr.algorithm.error)
        plot.plot(lr.algorithm.error, c='green', label="Error de entrenamiento")
        plot.plot(lr.algorithm.validationError, label="Error sobre validación")
        plot.legend()
        plot.show()

        y_predict = lr.predict(x_test)
        print("[test_ejercicio4] MSE", MSE()(y_test, y_predict))

        plot.figure()
        plot.plot(x_test,y_test)
        plot.plot(x_test,y_predict)

if __name__ == '__main__':
    unittest.main()
