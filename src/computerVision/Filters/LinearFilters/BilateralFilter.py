import numpy as np
import cv2 as cv

from src.computerVision.IFilter import IFilter


class BilateralFilter(IFilter):

    def __init__(self, size):
        super().__init__()
        self.size = size

    def apply(self, img: np.ndarray) -> np.ndarray:
        d = self.size
        return cv.bilateralFilter(img, d, 2 * d, d / 2)
