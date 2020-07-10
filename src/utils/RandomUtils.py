import numpy as np


class RandomUtils:
    @staticmethod
    def exp(lambda_param, samples) -> np.ndarray:
        uniform = np.random.uniform(0, 1, samples)
        return -(np.log(1 - uniform)) / lambda_param
