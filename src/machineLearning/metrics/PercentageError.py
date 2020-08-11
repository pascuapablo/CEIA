import numpy as np

from src.machineLearning.metrics.IMetric import IMetric


class PercentageError(IMetric):

    def __call__(self, target, prediction):
        return np.sum(prediction == target) / len(target) * 100
