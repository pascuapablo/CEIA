import numpy as np

from src.machineLearning.metrics.IMetric import IMetric


class Precision(IMetric):

    def __call__(self, target, prediction):
        TP = np.sum((target == 1) & (prediction == 1))
        FP = np.sum((target == 1) ^ (prediction == 1))
        return TP / (TP + FP) * 100
