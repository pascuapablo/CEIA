import unittest

import matplotlib.pyplot as plot
import numpy as np
from sklearn.neighbors import KernelDensity


class Ejercicio1(unittest.TestCase):
    def test_ejercicio(self):
        p = 0.01
        epochs = 3000
        realizaciones = np.zeros((epochs, 300))
        for samples in range(1, 300):
            for i in range(epochs):
                y = np.random.uniform(0, 1, samples)
                one_indexes = y <= p
                zero_indexes = y > p

                y[one_indexes] = 1
                y[zero_indexes] = 0

                cantidad_de_fallos = np.sum(y)
                realizaciones[i, samples] = cantidad_de_fallos

        # marco como 1 las realizaciones que tuvieron 1 o mas taza de fallos
        realizaciones[realizaciones >= 1] = 1
        prob_falla = np.mean(realizaciones, axis=0)

        fig = plot.figure()
        ax = fig.add_subplot(111)
        ax.plot(prob_falla, label="Probabilidad de 1 o mas fallos")
        ax.plot(np.ones(300) * 0.5, '--', c="red", label="P(Y>=1) = 0.5")
        plot.xlabel("Cantidad de fósforos fabricados")
        plot.legend()
        plot.show()


if __name__ == '__main__':
    unittest.main()
