import numpy as  np
import cv2 as cv

from src.computerVision.IFilter import IFilter


class GaussianBlurFilter(IFilter):

    def __init__(self, size: int = 5, sigma=2) -> None:
        super().__init__()
        self.size = (size, size)
        self.sigma = sigma

    def apply(self, img: np.ndarray) -> np.ndarray:
        return cv.GaussianBlur(img, ksize=self.size, sigmaX=self.sigma)
