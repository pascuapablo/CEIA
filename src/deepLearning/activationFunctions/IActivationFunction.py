from abc import abstractmethod

import numpy as np


class IActivationFunction(object):
    def __init__(self):
        pass

    @abstractmethod
    def apply(self, z: np.ndarray) -> np.ndarray:
        raise NotImplemented

    @abstractmethod
    def apply_over_derivative(self, z: np.ndarray) -> np.ndarray:
        raise NotImplemented
