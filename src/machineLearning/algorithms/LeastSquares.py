import numpy as np

from src.machineLearning.algorithms.ILinearAlgorithm import ILinearAlgorithm
from src.machineLearning.algorithms.prediciontFunctions.LinealPrediction import LinearPrediction


class LeastSquares(ILinearAlgorithm):
    def __init__(self):
        super().__init__()
        self.prediction_function = LinearPrediction()

    def run(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.linalg.inv(x.T @ x) @ x.T @ y
