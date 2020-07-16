from src.machineLearning.metrics.IMetric import IMetric
import numpy as np


class MeanError(IMetric):

    def __call__(self, target, prediction):
        return np.mean((target - prediction))
