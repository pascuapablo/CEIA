import unittest

import numpy as np


class Ejercicio2(unittest.TestCase):
    def test_ejercicio(self):
        n_simulaciones = 30000
        N = 10
        mu = 48
        sigma = 4

        realizaciones = np.random.normal(mu, sigma / np.sqrt(N), (n_simulaciones, N))
        min_teorico = mu - 1.96 * sigma / np.sqrt(N)
        max_teorico = mu + 1.96 * sigma / np.sqrt(N)

        muestras_en_el_intervalo = realizaciones[(realizaciones > min_teorico) & (
                realizaciones < max_teorico)]
        confianza = len(muestras_en_el_intervalo) / (n_simulaciones * N)
        print(" Confianza simulada ", confianza)
        # Confianza simulada 0.94994
