import numpy as np

from src.machineLearning.algorithms.ILinearAlgorithm import ILinearAlgorithm
from src.machineLearning.algorithms.prediciontFunctions.IPredictionFunction import IPredictionFunction
from src.machineLearning.algorithms.prediciontFunctions.LinealPrediction import LinearPrediction


class MiniBatchGradientDescent(ILinearAlgorithm):

    def __init__(self, learning_rate: float = 0.01, n_epochs: int = 100, n_batches: int = 16,
                 prediction_function: IPredictionFunction = LinearPrediction()):
        super().__init__()
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.prediction_function = prediction_function
        self.n_batches = n_batches

    def run(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
          shapes:
              X_t = nxm
              y_t = nx1
              W = mx1
          """

        n = x.shape[0]
        m = x.shape[1]

        x_validation = x[round(n / 5):round(n * 4 / 5)]
        x_train = x[0:round(n / 5)]

        y_validation = y[round(n / 5):round(n * 4 / 5)]
        y_train = y[0:round(n / 5)]

        # initialize random weights
        W = np.random.randn(m).reshape(m, 1)

        batch_size = int(len(x_train) / self.n_batches)

        self.error = np.zeros(self.n_epochs)
        self.validationError = np.zeros(self.n_epochs)

        for i in range(self.n_epochs):
            idx = np.random.permutation(x_train[0:(batch_size * self.n_batches)].shape[0])
            x_epoch = x_train[idx]
            y_epoch = y_train[idx]

            for j in range(0, len(x_epoch), batch_size):
                end = j + batch_size if j + batch_size <= len(x_epoch) else len(x_epoch)
                batch_X = x_epoch[j: end]
                batch_y = y_epoch[j: end]

                # print("[run] W", W)
                # print("[run] x", batch_X)
                prediction = self.prediction_function.predict(W, batch_X)  # nx1
                # print("[run] prediciont", prediction)

                error = prediction - batch_y  # nx1
                # print(error)
                # print("[run] error", error.shape)

                self.error[i] = np.sqrt(np.sum(error ** 2, axis=0)) / batch_size
                self.validationError[i] = np.sqrt(
                    np.sum((self.prediction_function.predict(W, x_validation) - y_validation) ** 2,
                           axis=0)) / len(x_validation)

                loss = error * batch_X
                # print("[run] loss", loss.shape)
                gradient = self.prediction_function.gradient_scalar_factor(x_epoch) * np.sum(loss, axis=0,
                                                                                             keepdims=True)  # mx1

                # print("[run] gradient", gradient.shape)
                W = W - (self.lr * gradient.T)
                # print("[run] w",W.shape)

        # self.error = np.mean(np.sqrt(np.sum(self.error ** 2, axis=0) / batch_size), axis=0)
        return W
