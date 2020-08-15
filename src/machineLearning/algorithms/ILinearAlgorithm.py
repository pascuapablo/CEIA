import numpy as np

from src.machineLearning.algorithms.prediciontFunctions.IPredictionFunction import IPredictionFunction


class ILinearAlgorithm(object):
    def __init__(self):
        self.error = None
        self.validationError = None
        self.prediction_function: IPredictionFunction = None

    def run(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError
