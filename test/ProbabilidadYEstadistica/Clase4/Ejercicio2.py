import unittest
import numpy as np


class Ejercicio2(unittest.TestCase):
    def test_something(self):
        # 1=seca y 0=cara
        X = np.random.uniform(0, 1, 10000).round()

        p_maxima_verosimilitud = np.mean(X)
        print("Estimador p de maxima verosimilitud: ", p_maxima_verosimilitud)
        # Estimador p de maxima verosimilitud:  0.5076


if __name__ == '__main__':
    unittest.main()
