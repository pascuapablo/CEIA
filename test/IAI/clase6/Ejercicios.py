import unittest

import matplotlib.pyplot as plot
import numpy as np

from src.machineLearning.algorithms.MiniBatchGradientDescent import MiniBatchGradientDescent
from src.machineLearning.algorithms.StochasticGradientDescent import StochasticGradientDescent
from src.machineLearning.algorithms.prediciontFunctions.SigmoidPrediction import SigmoidPrediction
from src.machineLearning.classification.LinearClassification import LinearClassfication
from src.machineLearning.metrics.PercentageError import PercentageError
from src.machineLearning.metrics.QuantityError import QuantityError
from src.utils.DatasetUtils import DatasetUtils


class MyTestCase(unittest.TestCase):
    def test_something(self):
        ds = DatasetUtils(path="clase_6_dataset.txt")
        ds_train, ds_test = ds.split(80)

        x_train = ds_train[:, 0:2]
        y_train = ds_train[:, 2:3]

        lc = LinearClassfication()
        lc.fit(x_train, y_train)

        lc_sgd = LinearClassfication(algorithm=StochasticGradientDescent(learning_rate=0.015,
                                                                         n_epochs=23000,
                                                                         prediction_function=SigmoidPrediction()))
        lc_sgd.fit(x_train, y_train)

        lc_mbgd = LinearClassfication(algorithm=MiniBatchGradientDescent(learning_rate=0.001,
                                                                         n_epochs=100000,
                                                                         n_batches=16,
                                                                         prediction_function=SigmoidPrediction()))
        lc_mbgd.fit(x_train, y_train)

        x_test = ds_test[:, 0:2]
        y_test = ds_test[:, 2:3]

        y_predict_gd = lc.predict(x_test)
        y_predict_sgd = lc_sgd.predict(x_test)
        y_predict_mbgd = lc_mbgd.predict(x_test)

        percentage = PercentageError()
        quantity = QuantityError()
        print("Aciertos GD %: ", percentage(y_test, y_predict_gd), "%")
        print("Error GD (cantidad de fallos GD): ",  quantity(y_test, y_predict_gd))

        print("Aciertos SGD %: ", percentage(y_test, y_predict_sgd), "%")
        print("Error SGD (cantidad de fallos GD): ", quantity(y_test, y_predict_sgd))

        print("Aciertos MBGD %: ", percentage(y_test, y_predict_mbgd), "%")
        print("Error MBGD (cantidad de fallos GD): ", quantity(y_test, y_predict_mbgd))



        plot.figure()
        ones_index = (y_test == 1)[:, 0]
        zeros_index = (y_test == 0)[:, 0]
        plot.plot(x_test[ones_index, 0], x_test[ones_index, 1], 'o')
        plot.plot(x_test[zeros_index, 0], x_test[zeros_index, 1], 'x')

        x = np.linspace(-0, 200, len(x_train))
        y = -lc.model[1] / lc.model[0] * x - lc.model[2] / lc.model[0] + np.log(1)
        y_sgd = -lc_sgd.model[1] / lc_sgd.model[0] * x - lc_sgd.model[2] / lc_sgd.model[0] + np.log(1)
        y_mbgd = -lc_mbgd.model[1] / lc_mbgd.model[0] * x - lc_mbgd.model[2] / lc_mbgd.model[0] + np.log(1)

        plot.plot(x, y, c='green', label='Modelo gradiente descendiente')
        plot.plot(x, y_sgd, c='blue', label='Modelo gradiente descendiente estocastico')
        plot.plot(x, y_mbgd, c='red', label='Modelo mini batch gradiente descendiente')
        plot.legend()

        plot.figure()
        ones_index = (y_train == 1)[:, 0]
        zeros_index = (y_train == 0)[:, 0]
        plot.plot(x_train[ones_index, 0], x_train[ones_index, 1], 'o')
        plot.plot(x_train[zeros_index, 0], x_train[zeros_index, 1], 'x')

        x = np.linspace(-0, 200, len(x_train))
        y = -lc.model[1] / lc.model[0] * x - lc.model[2] / lc.model[0] + np.log(1)
        y_sgd = -lc_sgd.model[1] / lc_sgd.model[0] * x - lc_sgd.model[2] / lc_sgd.model[0] + np.log(1)
        y_mbgd = -lc_mbgd.model[1] / lc_mbgd.model[0] * x - lc_mbgd.model[2] / lc_mbgd.model[0] + np.log(1)
        plot.plot(x, y, c='green', label='Modelo gradiente descendiente')
        plot.plot(x, y_sgd, c='blue', label='Modelo gradiente descendiente estocastico')
        plot.plot(x, y_mbgd, c='red', label='Modelo mini batch gradiente descendiente ')
        plot.legend()

        plot.figure()
        plot.plot(lc.algorithm.error, c='green', label="gd")
        plot.plot(lc_sgd.algorithm.error, c='blue', label="sgd")
        plot.plot(lc_mbgd.algorithm.error, c='red', label="mbgd")
        plot.legend()
        plot.show()


if __name__ == '__main__':
    unittest.main()
