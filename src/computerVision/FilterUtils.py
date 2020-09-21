from typing import List

import numpy as np

from src.computerVision.IFilter import IFilter


class FilterUtils:

    def __init__(self, img: np.ndarray) -> None:
        self.img = img

    def apply_all(self, filters: List[IFilter]):
        out = self.img
        for f in filters:
            out = f.apply(out)
        return out
