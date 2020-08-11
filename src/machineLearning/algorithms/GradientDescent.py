import numpy as np

from src.machineLearning.algorithms.ILinearAlgorithm import ILinearAlgorithm
from src.machineLearning.algorithms.prediciontFunctions.IPredictionFunction import IPredictionFunction
from src.machineLearning.algorithms.prediciontFunctions.LinealPrediction import LinearPrediction


class GradientDescent(ILinearAlgorithm):

    def __init__(self, learning_rate: float = 0.01, n_epochs: int = 100,
                 prediction_function: IPredictionFunction = LinearPrediction()):
        super().__init__()
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.prediction_function = prediction_function

    def run(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
         shapes:
             X_t = nxm
             y_t = nx1
             W = mx1
         """
        # initialize random weights
        m = x.shape[1]
        W = np.random.randn(m).reshape(m, 1)
        self.error = np.zeros(self.n_epochs)
        for i in range(self.n_epochs):
            prediction = self.prediction_function.predict(W, x)
            error = prediction - y
            self.error[i] = np.sqrt(np.sum(error ** 2)) / len(error)
            loss = error * x
            gradient = self.prediction_function.gradient_scalar_factor(x) * np.sum(loss, axis=0, keepdims=True)
            W = W - (self.lr * gradient.T)

        return W
