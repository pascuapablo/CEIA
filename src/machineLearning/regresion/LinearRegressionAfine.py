import numpy as np

from src.machineLearning.IMLBaseModel import IMLBaseModel


class LinearRegressionAffine(IMLBaseModel):

    def __init__(self, order: int = 1):
        super().__init__()
        self.bias = 0
        self.order = order

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x_train = x
        y_train = y
        if x.ndim == 1:
            x_train = x[:, None]

        if y.ndim == 1:
            y_train = y[:, None]

        x_train = self.__reshape_x(x_train)
        w = np.linalg.inv(x_train.T @ x_train) @ x_train.T @ y_train

        self.model: np.ndarray = w
        self.bias = w[-1:, :]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        :rtype: np.ndarray
        """
        x_to_predict = x
        if x.ndim == 1:
            x_to_predict = x[:, None]

        x_to_predict = self.__reshape_x(x_to_predict)

        if self.model.ndim == 1 or (self.model.shape[0] == 1 and self.model.shape[1] == 1):
            return x_to_predict * self.model + self.bias
        else:
            return np.matmul(self.model.T, x_to_predict.T).T

    def __reshape_x(self, x: np.ndarray) -> np.ndarray:
        result: np.ndarray = x
        for i in range(1, self.order):
            result = np.append(result, result[:, 0:1] ** (i + 1), axis=1)

        result = np.append(result, np.ones((x.shape[0], 1)), axis=1)

        return result
