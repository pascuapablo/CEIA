import numpy as np

from src.computerVision.IFilter import IFilter


class WhitePatchFilter(IFilter):

    def apply(self, img: np.ndarray) -> np.ndarray:
        rgmMax = np.max(np.max(img, axis=0, keepdims=True), axis=1, keepdims=True)
        rgmMax[rgmMax == 0] = 1
        max_normalized = 255 / rgmMax

        return np.uint8(np.multiply(img, max_normalized))
