import numpy as np

from src.computerVision.Filters.LinearFilters.GaussianBlurFilter import GaussianBlurFilter
from src.computerVision.IFilter import IFilter


class UnsharpMaskFilter(IFilter):
    def __init__(self, k, filter_size=(5, 5), sigma=2):
        super().__init__()
        self.k = k
        self.gfilter = GaussianBlurFilter(filter_size, sigma)

    def apply(self, img: np.ndarray) -> np.ndarray:
        return (self.k + 1) * img - self.k * self.gfilter.apply(img)
