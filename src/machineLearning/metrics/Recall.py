import numpy as np

from src.machineLearning.metrics.IMetric import IMetric


class Recall(IMetric):

    def __call__(self, target, prediction):
        TP = np.sum((prediction == 1) & (target == 1))
        P = np.sum(target)
        return TP / P * 100
