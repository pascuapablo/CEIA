import numpy as np
import cv2 as cv

from src.computerVision.IFilter import IFilter


class SobelYFilter(IFilter):
    def __init__(self):
        super().__init__()
        self.kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, 1]])

    def apply(self, img: np.ndarray) -> np.ndarray:
        return cv.filter2D(img, -1, self.kernel)
