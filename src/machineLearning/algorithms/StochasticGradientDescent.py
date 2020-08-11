import numpy as np

from src.machineLearning.algorithms.ILinearAlgorithm import ILinearAlgorithm
from src.machineLearning.algorithms.prediciontFunctions.IPredictionFunction import IPredictionFunction
from src.machineLearning.algorithms.prediciontFunctions.LinealPrediction import LinearPrediction


class StochasticGradientDescent(ILinearAlgorithm):

    def __init__(self, learning_rate: float = 0.01, n_epochs: int = 100,
                 prediction_function: IPredictionFunction = LinearPrediction()):
        super().__init__()
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.prediction_function = prediction_function

    def run(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        m = x.shape[1]
        # initialize random weights
        W = np.random.randn(m).reshape(m, 1)
        self.error = np.zeros((self.n_epochs, n))
        for i in range(self.n_epochs):
            idx = np.random.permutation(x.shape[0])
            x = x[idx]
            y = y[idx]
            # print("[run] x",x.shape)
            # print("[run] y",y.shape)
            for j in range(n):
                prediction = self.prediction_function.predict(W, x[j:j + 1, :])
                # print("[run] predictin", prediction.shape)
                error = prediction - y[j, :]  # 1x1
                # print("[run] error", error.shape)
                self.error[i, j] = error
                loss = error * x[j, :]
                # print("[run] loss", loss.shape)
                gradient = self.prediction_function.gradient_scalar_factor(x) * loss
                # print("[run] gradient", gradient.shape)
                W = W - (self.lr * gradient.T)
                # print("[run] w", W.shape)
        self.error = np.sqrt(np.sum(self.error ** 2, axis=1)) / self.error.shape[0]
        return W
