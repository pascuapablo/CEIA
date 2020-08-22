import unittest

import matplotlib.pyplot as plot
import numpy as np
from sklearn.neighbors import KernelDensity


class Ejercicio2(unittest.TestCase):
    def test_ejercicio2c(self):
        n_samples = 10
        u = np.random.uniform(0, 1, n_samples)
        # Simulacion de muestras usando el metodo de la transformada inversa
        y = 3 * np.sqrt(u)
        print("Simulaciones", y)
        # Simulaciones [2.45633313 2.09930483 1.44194413 2.70517394 2.89434844 2.34250903  0.95256784 1.698109
        # 2.56783211 2.79402629]

    def test_ejercicio2d(self):
        n_samples = 10000
        u = np.sort(np.random.uniform(0, 1, n_samples))
        u = u[:, np.newaxis]

        y = 3 * np.sqrt(u)

        fig, axs = plot.subplots(3, 1)

        kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(y)
        log_dens = kde.score_samples(y)
        axs[0].plot(u * 3, np.exp(log_dens))
        axs[0].text(0.2, 0.5, "ancho de banda = 0.05")

        kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(y)
        log_dens = kde.score_samples(y)
        axs[1].plot(u * 3, np.exp(log_dens))
        axs[1].text(0.2, 0.5, "ancho de banda = 0.1")

        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(y)
        log_dens = kde.score_samples(y)
        axs[2].plot(u * 3, np.exp(log_dens))
        axs[2].text(0.2, 0.5, "ancho de banda = 0.2")

        plot.show()


if __name__ == '__main__':
    unittest.main()
