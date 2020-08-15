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
        minErrorOrder = -1

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
                minErrorOrder = i

        print("[test_something] El orden del modelo con menor error fue el ", minErrorOrder)
        # [test_something] El orden del modelo con menor error fue el  3

        bestModel = models[minErrorOrder - 1]
        print("[test_ejercicio3] Mejor modelo", bestModel.model)
        print("[test_ejercicio3] Mejor modelo MSE", error(target=y_test, prediction=bestModel.predict(x_test)))
        plot.figure()
        plot.scatter(x_test, y_test, c='red', label="target")
        plot.scatter(x_test, bestModel.predict(x_test), c='blue', label="prediccion")

        x_axis = np.linspace(-400, 400, 1000)
        x = x_axis[:, None]
        for i in range(1, minErrorOrder):
            x = np.append(x, x[:, 0:1] ** (i + 1), axis=1)
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)

        model = (bestModel.model.T @ x.T).T
        plot.plot(x_axis, model, c='green', label="modelo")
        plot.legend()
        plot.show()

    def test_ejercicio4(self):
        ds = DatasetUtils(path="clase_8_dataset.csv")
        ds_train, ds_test = ds.split(80)

        x_train = ds_train[:, 0:1]
        y_train = ds_train[:, 1:2]

        x_test = ds_test[:, 0:1]
        y_test = ds_test[:, 1:2]

        order = 3
        lr = LinearRegressionAffine(order=order,
                                    algorithm=MiniBatchGradientDescent(learning_rate=0.00000000000003,
                                                                       n_epochs=30000,
                                                                       n_batches=30))

        lr.fit(x_train, y_train)

        plot.figure()
        print("[test_ejercicio4] error", lr.algorithm.error)
        plot.plot(lr.algorithm.error, c='green', label="Error de entrenamiento")
        plot.plot(lr.algorithm.validationError, label="Error sobre validación")
        plot.ylim(top=1000, bottom=0)
        plot.legend()

        y_predict = lr.predict(x_test)
        print("[test_ejercicio4] MSE", MSE()(y_test, y_predict))

        plot.figure()
        plot.scatter(x_test, y_test, c='red', label="target")
        plot.scatter(x_test, y_predict, c='blue', label="prediccion")

        x_axis = np.linspace(-400, 400, 1000)
        x = x_axis[:, None]
        for i in range(1, order):
            x = np.append(x, x[:, 0:1] ** (i + 1), axis=1)
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)

        model = (lr.model.T @ x.T).T
        plot.plot(x_axis, model, c='green', label="modelo")
        plot.legend()
        plot.show()

        # Si se compara el modelo obtenido con mini batch vs el deterministico de cuadrados minimos, se puede ver como
        # en cuadrados minimos el fit es casi perfecto, mientras que en mini Batch si bien el modelo toma "la forma"
        # de la curva, tiene cierto desvio. De hecho si vamos a los numeros, el MSE de cuadrados minimos fue de 12
        # mientras que el de mini batch es de 153 es decir un orden de magnitud mas grande.
        # Es probable tambien que esta diferencia tan grande sea debido a que los valores de x eran grandes y hacia que
        # el algoritmo diverja muy rápido


if __name__ == '__main__':
    unittest.main()
