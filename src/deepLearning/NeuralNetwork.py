from os import system
from typing import Union, List

import numpy as np

from src.deepLearning.Layer import Layer
from src.deepLearning.algorithms.BackPropagationMiniBatch import BackPropagationMiniBatch
from src.deepLearning.algorithms.IBackPropagation import IBackPropagation
from src.machineLearning.IMLBaseModel import IMLBaseModel
from src.utils.DatasetUtils import DatasetUtils


def checkSizes(z_s):
    for i, z in enumerate(z_s):
        print("layer", i, "z shape", z.shape)


class NeuralNetwork(IMLBaseModel):

    def __init__(self, layers: List[Layer] = [], algorithm: IBackPropagation = BackPropagationMiniBatch()):
        super().__init__()
        self.algorithm = algorithm
        self.algorithm.setLayers(layers)

    def fit(self, x: np.ndarray, y: np.ndarray, n_epochs: int = 40, learning_rate: float = 0.01,
            batch_size: int = 32):

        y_shaped = y
        if y.ndim == 1:
            y_shaped = y[:, None]

        train, validation = DatasetUtils(np.concatenate((x, y_shaped), axis=1)).split(88)

        x_train = train[:, 0:x.shape[1]]
        y_train = train[:, x.shape[1]:]

        x_validation = validation[:, 0:x.shape[1]]
        y_validation = validation[:, x.shape[1]:]

        n_batches = len(y_train) // batch_size
        train_error = np.empty((n_epochs, n_batches, batch_size))
        validation_error = np.empty((n_epochs, n_batches, len(x_validation)))
        for epoch in range(n_epochs):
            for i in range(0, batch_size * n_batches, batch_size):
                x_batch = x_train[i:(i + batch_size)]
                y_batch = y_train[i:(i + batch_size)]

                # Trainning
                (z_s, a_s) = self.algorithm.forward(x_batch)

                y_predict = a_s[-1]
                batch_error = -2 * (y_batch - y_predict)
                train_error[epoch, int(i / batch_size), :] = batch_error[:, 0]

                # Validation
                (_, a_validation_s) = self.algorithm.forward(x_validation)
                batch_validation_error = -y_validation + a_validation_s[-1]
                validation_error[epoch, int(i / batch_size), :] = batch_validation_error[:, 0]

                (delta_w, delta_b) = self.algorithm.backwards(x_batch, z_s, batch_error)
                self.algorithm.update(delta_w, delta_b, learning_rate)

        train_error = np.sum(np.sum(train_error, axis=2), axis=1) / (train_error.shape[2] * train_error.shape[1])
        validation_error = np.sum(np.sum(validation_error, axis=2), axis=1) / (
                validation_error.shape[2] * validation_error.shape[1])

        return train_error, validation_error

    def predict(self, x) -> np.ndarray:
        _, a = self.algorithm.forward(x)
        return np.round(a[-1])
