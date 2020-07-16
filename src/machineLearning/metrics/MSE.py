from src.machineLearning.metrics.IMetric import IMetric
import numpy as np


class MSE(IMetric):
    def __call__(self, target, prediction):
        return np.mean((target - prediction) ** 2)
