import numpy as np
from numpy.core._multiarray_umath import ndarray

from src.computerVision.IFilter import IFilter


class RemoveZeros(IFilter):
    def apply(self, img: np.ndarray) -> np.ndarray:
        newImage: ndarray = img
        newImage[newImage == 0] = 1
        return newImage

    def __init__(self):
        super().__init__()
