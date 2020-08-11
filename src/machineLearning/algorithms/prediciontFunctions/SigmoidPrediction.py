import numpy as np

from src.machineLearning.algorithms.prediciontFunctions.IPredictionFunction import IPredictionFunction


class SigmoidPrediction(IPredictionFunction):
    def gradient_scalar_factor(self, x: np.ndarray) -> float:
        return 1 / len(x)

    def predict(self, w: np.ndarray, x: np.ndarray) -> np.ndarray:
        if w.ndim == 1 or (w.shape[0] == 1 and w.shape[1] == 1):
            wTimesX = x * w
        else:
            wTimesX = -x @ w

        return 1 / (1 + np.exp(wTimesX))
