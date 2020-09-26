import numpy as np

from src.deepLearning.activationFunctions.IActivationFunction import IActivationFunction


class SigmoidActivationFunction(IActivationFunction):
    def __init__(self):
        super().__init__()

    def apply(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def apply_over_derivative(self, z: np.ndarray) -> np.ndarray:
        a = self.apply(z)
        return np.multiply(a, (1 - a))
