from abc import abstractmethod
import numpy as np


class IFilter(object):

    @abstractmethod
    def apply(self, img: np.ndarray) -> np.ndarray:
        raise NotImplemented
