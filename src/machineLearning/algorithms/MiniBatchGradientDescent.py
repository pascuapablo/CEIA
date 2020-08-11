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

        # initialize random weights
        W = np.random.randn(m).reshape(m, 1)

        batch_size = int(len(x) / self.n_batches)
        self.error = np.zeros((batch_size, int(len(x) / batch_size), self.n_epochs))

        for i in range(self.n_epochs):
            idx = np.random.permutation(x[0:(batch_size * self.n_batches)].shape[0])
            x = x[idx]
            y = y[idx]

            for j in range(0, len(x), batch_size):
                end = j + batch_size if j + batch_size <= len(x) else len(x)
                batch_X = x[j: end]
                batch_y = y[j: end]

                prediction = self.prediction_function.predict(W, batch_X)  # nx1

                error = prediction - batch_y  # nx1
                # print("[run] errpr", error.shape)
                # print("[run] self error",  self.error[:, j, i:i+1].shape)
                self.error[:, int(j / batch_size), i:i + 1] = error
                loss = error * batch_X

                gradient = self.prediction_function.gradient_scalar_factor(x) * np.sum(loss, axis=0,
                                                                                       keepdims=True)  # mx1
                W = W - (self.lr * gradient.T)

        self.error = np.mean(np.sqrt(np.sum(self.error ** 2, axis=0)) / batch_size, axis=0)
        return W
