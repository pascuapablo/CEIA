import numpy as np

from src.computerVision.IFilter import IFilter


class ChromaticCoordinatesFilter(IFilter):

    def apply(self, img: np.ndarray) -> np.ndarray:
        sum_channels = np.sum(img, axis=2, keepdims=True)
        sum_channels[sum_channels == 0] = 1
        img_codenadas_crom = np.uint8((img / sum_channels) * 255)
        return img_codenadas_crom
