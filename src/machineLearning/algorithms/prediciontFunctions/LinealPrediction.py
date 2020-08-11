import numpy as np

from src.machineLearning.algorithms.prediciontFunctions.IPredictionFunction import IPredictionFunction


class LinearPrediction(IPredictionFunction):
    def predict(self, w: np.ndarray, x: np.ndarray) -> np.ndarray:
        if w.ndim == 1 or (w.shape[0] == 1 and w.shape[1] == 1):
            return x * w
        else:
            return np.matmul(x, w)

    def gradient_scalar_factor(self, x: np.ndarray) -> float:
        return 2 / len(x)
