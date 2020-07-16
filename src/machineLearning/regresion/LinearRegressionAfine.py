from src.machineLearning.IMLBaseModel import IMLBaseModel
import numpy as np


class LinearRegressionAffine(IMLBaseModel):

    def __init__(self):
        super().__init__()
        self.bias = 0

    def fit(self, x, y):
        x_train = x
        y_train = y
        if x.ndim == 1:
            x_train = x[:, None]

        if y.ndim == 1:
            y_train = y[:, None]

        x_train = np.append(x_train, np.ones((x_train.shape[0], 1)), axis=1)
        w = np.linalg.inv(x_train.T @ x_train) @ x_train.T @ y_train

        self.model: np.ndarray = w[0:-1, :]
        self.bias = w[-1:, :]

    def predict(self, x):
        x_to_predict = x
        if x.ndim == 1:
            x_to_predict = x[:, None]

        if self.model.ndim == 1 or (self.model.shape[0] == 1 and self.model.shape[1] == 1):
            return x_to_predict * self.model + self.bias
        else:
            return np.matmul(self.model.T, x_to_predict)
