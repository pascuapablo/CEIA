import numpy as  np
import cv2 as cv

from src.computerVision.IFilter import IFilter


class GaussianBlurFilter(IFilter):

    def __init__(self, size, sigma) -> None:
        super().__init__()
        self.size = size
        self.sigma = sigma

    def apply(self, img: np.ndarray) -> np.ndarray:
        return cv.GaussianBlur(img, ksize=self.size, sigmaX=self.sigma)
