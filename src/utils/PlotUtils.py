from typing import List

import matplotlib
import numpy as np

from src.machineLearning.IMLBaseModel import IMLBaseModel


class PlotUtils:
    def __init__(self):
        pass

    def plotDecisionBoundry(self, x_test: np.ndarray, decision_model: IMLBaseModel, axis):
        x_min, x_max = x_test[:, 0].min() - .5, x_test[:, 0].max() + .5
        y_min, y_max = x_test[:, 1].min() - .5, x_test[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                             np.arange(y_min, y_max, 0.2))
        predictions = decision_model.predict(np.c_[xx.ravel(), yy.ravel()])
        axis.contourf(xx, yy, predictions.reshape(xx.shape),
                      cmap=matplotlib.colors.ListedColormap(["blue", "red"]),
                      alpha=0.2)

    def plotHistory(self, history, metrics: List, plt):
        figs = []
        metricsNames = [m.name for m in metrics]
        metricsNames.append("loss")
        for metric in metricsNames:
            fig = plt.figure()
            fig.suptitle(metric)
            plt.plot(history.history[metric], label='Train')
            plt.plot(history.history["val_" + metric], label='Validation')
            plt.legend()
