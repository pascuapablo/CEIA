import numpy as np


class IPredictionFunction(object):
    def predict(self, w: np.ndarray, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def gradient_scalar_factor(self, x: np.ndarray) -> float:
        raise NotImplementedError
