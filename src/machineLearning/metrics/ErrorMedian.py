from src.machineLearning.metrics.IMetric import IMetric
import numpy as np


class MSE(IMetric):
    def __call__(self, target, prediction):
        return np.sqrt(np.sum((target - prediction) ** 2))
