from abc import abstractmethod
from typing import List, Any

import numpy as np

from src.deepLearning.Layer import Layer


class IBackPropagation(object):

    def __init__(self):
        super().__init__()
        self.w = []
        self.b = []
        self.layers: List[Layer] = []

    def setLayers(self, layers: List[Layer]) -> None:
        self.layers = layers

    @abstractmethod
    def forward(self, x_train: np.ndarray):
        raise NotADirectoryError

    @abstractmethod
    def backwards(self, x: np.ndarray, z: List[np.ndarray], error: np.ndarray):
        raise NotADirectoryError

    @abstractmethod
    def update(self, delta_w: List[np.ndarray], delta_b: List[np.ndarray], learning_rate: float):
        raise NotADirectoryError
