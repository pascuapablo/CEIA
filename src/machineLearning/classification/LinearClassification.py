from typing import Union

import numpy as np

from src.machineLearning.IMLBaseModel import IMLBaseModel
from src.machineLearning.algorithms.GradientDescent import GradientDescent
from src.machineLearning.algorithms.ILinearAlgorithm import ILinearAlgorithm
from src.machineLearning.algorithms.prediciontFunctions.SigmoidPrediction import SigmoidPrediction


class LinearClassfication(IMLBaseModel):
    def __init__(self,
                 order: int = 1,
                 algorithm: ILinearAlgorithm = GradientDescent(learning_rate=0.0012,
                                                               n_epochs=600000,
                                                               prediction_function=SigmoidPrediction())):
        super().__init__()
        self.bias = 0
        self.order = order
        self.algorithm = algorithm

    def fit(self, x: np.ndarray, y: Union[None, np.ndarray]):
        x_train = x
        y_train = y
        if x.ndim == 1:
            x_train = x[:, None]

        if y.ndim == 1:
            y_train = y[:, None]

        x_train = self.__add_new_orders(x_train)

        w = self.algorithm.run(x_train, y_train)
        self.model: np.ndarray = w
        self.bias = w[-1:, :]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        :rtype: np.ndarray
        """
        x_to_predict = x
        if x.ndim == 1:
            x_to_predict = x[:, None]

        x_to_predict = self.__add_new_orders(x_to_predict)

        return np.round(self.algorithm.prediction_function.predict(self.model, x_to_predict))

    def __add_new_orders(self, x: np.ndarray) -> np.ndarray:
        result: np.ndarray = x
        for i in range(1, self.order):
            result = np.append(result, result[:, 0:1] ** (i + 1), axis=1)

        result = np.append(result, np.ones((x.shape[0], 1)), axis=1)

        return result
