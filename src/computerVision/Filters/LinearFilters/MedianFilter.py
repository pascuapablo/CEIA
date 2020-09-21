import numpy as np
import cv2 as cv

from src.computerVision.IFilter import IFilter


class MedianFilter(IFilter):
    def __init__(self, size=5):
        super().__init__()
        self.size = size

    def apply(self, img: np.ndarray) -> np.ndarray:
        asd = cv.medianBlur(img, self.size)
        print("shape", asd.shape)
        return asd
