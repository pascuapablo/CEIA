import numpy as np

from src.computerVision.Filters.LinearFilters.GaussianBlurFilter import GaussianBlurFilter
from src.computerVision.IFilter import IFilter


class DoGFilter(IFilter):

    def __init__(self, low_sigma, high_sigma, size) -> None:
        super().__init__()
        self.lowSigma = GaussianBlurFilter(size=size, sigma=low_sigma)
        self.highSigma = GaussianBlurFilter(size=size, sigma=high_sigma)

    def apply(self, img: np.ndarray) -> np.ndarray:
        return self.lowSigma.apply(img) - self.highSigma.apply(img)
