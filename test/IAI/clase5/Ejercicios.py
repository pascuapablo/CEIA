import unittest

import matplotlib.pyplot as plot
import numpy as np

from src.machineLearning.metrics.MSE import MSE
from src.machineLearning.regresion.LinearRegressionAfine import LinearRegressionAffine
from src.utils.DatasetUtils import DatasetUtils


class MyTestCase(unittest.TestCase):
    def test_something(self):
        N_SAMPLES = 150
        NOISE = 1
        N_ORDERS = 10
        EPOCHS = 50

        x = np.linspace(0, 2 * np.pi, N_SAMPLES)[:, None]
        sin = np.sin(x)
        dataset = np.append(x, sin + np.random.normal(0, NOISE, N_SAMPLES)[:, None], axis=1)

        x_test: np.ndarray

        y_predictions = np.zeros(((N_SAMPLES * 0.2).__round__(), EPOCHS, N_ORDERS))
        all_x_test = np.zeros(((N_SAMPLES * 0.2).__round__(), N_ORDERS))
        lr_models = np.empty((EPOCHS, N_ORDERS), dtype=LinearRegressionAffine)
        errors = np.zeros((EPOCHS, N_ORDERS)) * np.nan

        for i in range(0, N_ORDERS):
            train, test = DatasetUtils(dataset).split(80)
            x_train = train[:, 0]
            y_train = train[:, 1]

            x_test = test[:, 0]
            y_test = test[:, 1]
            all_x_test[:, i] = x_test
            for j in range(0, EPOCHS):
                lr = LinearRegressionAffine(order=i + 1)
                lr.fit(x_train, y_train)
                lr_models[j, i] = lr

                y_predict = lr.predict(x_test)
                y_predictions[:, j, i:(i + 1)] = y_predict

                errors[j, i] = MSE().__call__(y_test[:, None], y_predict)

        order = np.argmin(np.nanmean(errors, axis=0))
        print("El menor MSE se da para el polinomio de grado ", order)

        target = all_x_test[:, order]
        targetAxisSorted = np.argsort(target)

        fig, axs = plot.subplots(2, 1)
        epochExample = 0
        axs[0].plot(y_predictions[targetAxisSorted, epochExample, order],
                    label="mejor prediccion orden " + str(order), )
        axs[0].plot(np.sin(np.sort(target)), 'o', label="target")
        axs[0].legend()

        axs[1].plot(np.nanmean(errors, axis=0), label="MSE con k-folds")
        plot.xlabel("Ordel del polinomio")
        axs[1].legend()
        plot.show()


if __name__ == '__main__':
    unittest.main()
