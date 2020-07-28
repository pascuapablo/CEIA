import unittest

import matplotlib.pyplot as plot
import numpy as np
from sklearn.neighbors import KernelDensity


class Ejercicio3(unittest.TestCase):
    def test_something(self):
        N = 1000
        X = np.random.normal(0, 1, N)

        fig, axs = plot.subplots(3, 1)
        axs[0].hist(X, bins=5)
        axs[1].hist(X, bins=20)
        axs[2].hist(X, bins=50)

        plot.show()

    def test_kernet(self):
        N = 1000
        X = np.random.normal(0, 1, N)[:, np.newaxis]
        X_plot = np.linspace(-5, 10, N)[:, np.newaxis]

        fig, axs = plot.subplots(3, 1)

        kde = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(X)
        log_dens = kde.score_samples(X_plot)
        axs[0].fill(X_plot[:, 0], np.exp(log_dens), )
        axs[0].text(4, 0.4, "ancho de banda = 0.01")

        kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(X)
        log_dens = kde.score_samples(X_plot)
        axs[1].fill(X_plot[:, 0], np.exp(log_dens), )
        axs[1].text(4, 0.3, "ancho de banda = 0.1")

        kde = KernelDensity(kernel='gaussian', bandwidth=0.6).fit(X)
        log_dens = kde.score_samples(X_plot)
        axs[2].fill(X_plot[:, 0], np.exp(log_dens), )
        axs[2].text(4, 0.25, "ancho de banda = 0.6")

        plot.show()


if __name__ == '__main__':
    unittest.main()
