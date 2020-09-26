from typing import Tuple, List, Any

import numpy as np

from src.deepLearning.algorithms.IBackPropagation import IBackPropagation


class BackPropagationMiniBatch(IBackPropagation):

    def __init__(self):
        super().__init__()

    def forward(self, x_train: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """

        :param x_train: size samples x input_size
        :return: A and Z  a list of length "layers amount", and in each position a matrix of size samples x neurons
        """
        a_all: List[np.ndarray] = [x_train]
        z_all: List[np.ndarray] = []

        for i, layer in enumerate(self.layers):
            z, a = layer.compute(a_all[i])
            a_all.append(a)
            z_all.append(z)
        return z_all, a_all

    def backwards(self, x: np.ndarray, z: List[np.ndarray], error: np.ndarray):
        """
        :param x:
        :param z: List of length len(layers). Each position has a samples x neurons_qty
        :param error: samples x 1
        :return: gradient of matrix W and b.
                W = input_size x neurons
                b = 1 x neurons
        """

        delta_z = [np.empty(1)] * len(self.layers)  # delta = dC/dZ  known as error for each layer
        # insert the last layer error
        delta_z[-1] = -2 * error * self.layers[-1].activation_function.apply(z[-1])
        # Perform BackPropagation
        for i in reversed(range(len(delta_z) - 1)):
            # print("z shape", delta_z[i + 1].shape)
            # print("w shape", self.layers[i + 1].w.T.shape)
            aux = delta_z[i + 1] @ self.layers[i + 1].w.T
            delta_z[i] = np.multiply(aux,
                                     self.layers[i].activation_function.apply_over_derivative(z[i]))

        batch_size = x.shape[0]

        dw = []  # dC/dW
        dw.append(x.T @ delta_z[0] / batch_size)
        for i in range(1, len(delta_z)):
            delta = delta_z[i]
            delta_w = self.layers[i - 1].activation_function.apply(delta_z[i - 1]).T @ delta
            dw.append(delta_w / batch_size)

        db = [np.sum(d, axis=0, keepdims=True) / float(batch_size) for d in delta_z]
        return dw, db

    def update(self, delta_w: List[np.ndarray], delta_b: List[np.ndarray], learning_rate: float):
        for i, layer in enumerate(self.layers):
            layer.w = layer.w + learning_rate * delta_w[i]
            layer.b = layer.b + learning_rate * delta_b[i]
