from typing import Union

import numpy as np


class MLBaseModel(object):

    def __init__(self):
        self.model = None

    def fit(self, x: np.ndarray, y: Union[None, np.ndarray]):
        # train del model

        raise NotImplementedError

    def predict(self, y):
        # return Y hat
        raise NotImplementedError
