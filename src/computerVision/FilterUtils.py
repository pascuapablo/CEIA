from typing import List

import numpy as np

from src.computerVision.IFilter import IFilter


class FilterUtils:
    def applyAll(self, img: np.ndarray, filters: List[IFilter]):
        out = img
        for f in filters:
            out = f.apply(out)
        return out
