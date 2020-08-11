import numpy as np

from src.machineLearning.metrics.IMetric import IMetric


class QuantityError(IMetric):

    def __call__(self, target, prediction):
        return np.sum(np.abs(prediction - target))
