import numpy as np

from src.machineLearning.IMLBaseModel import IMLBaseModel


class LinearRegression(IMLBaseModel):

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x_train = x
        if x.ndim == 1:
            x_train = x[:, None]

        w = np.linalg.inv(x_train.T @ x_train) @ x_train.T @ y

        self.model: np.ndarray = w

    def predict(self, x):
        x_to_predict = x
        if x.ndim == 1:
            x_to_predict = x[:, None]

        if self.model.ndim == 1:
            return x_to_predict * self.model
        else:
            return np.matmul(self.model.T, x_to_predict)
