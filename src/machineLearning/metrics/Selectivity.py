import numpy as np

from src.machineLearning.metrics.IMetric import IMetric


class Selectivity(IMetric):

    def __call__(self, target, prediction):
        TN = np.sum((target == 0) & (prediction == 0))
        N = np.sum(target == 0)
        return TN / N * 100
