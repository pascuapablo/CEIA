import numpy as np

from src.computerVision.Filters.LinearFilters.GaussianBlurFilter import GaussianBlurFilter
from src.computerVision.IFilter import IFilter


class DoGFilter(IFilter):

    def __init__(self, low_sigma=2, high_sigma=5, size=(5, 5)) -> None:
        super().__init__()
        self.lowSigma = GaussianBlurFilter(size=size, sigma=low_sigma)
        self.highSigma = GaussianBlurFilter(size=size, sigma=high_sigma)

    def apply(self, img: np.ndarray) -> np.ndarray:
        return self.lowSigma.apply(img) - self.highSigma.apply(img)
