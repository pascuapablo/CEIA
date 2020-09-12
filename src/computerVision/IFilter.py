import numpy as np


class IFilter(object):

    def apply(self, img: np.ndarray) -> np.ndarray:
        raise NotImplemented
