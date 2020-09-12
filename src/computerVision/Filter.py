import numpy as np
import typing


class Filter(object):

    def apply(self, img: np.ndarray) -> np.ndarray:
        raise NotImplemented

    @staticmethod
    def applyAll(entry_point: np.ndarray, filters: typing.List):
        out = entry_point
        for f in filters:
            out = f.apply(out)

        return out
