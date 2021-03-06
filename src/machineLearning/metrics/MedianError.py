from src.machineLearning.metrics.IMetric import IMetric
import numpy as np


class MedianError(IMetric):
    def __call__(self, target, prediction):
        return np.median((target - prediction))
