import unittest
import numpy as np


class Ejercicios(unittest.TestCase):
    def dice(self, samples: int = 1) -> np.ndarray:
        return np.random.uniform(1, 7, samples).astype(int)

    def test_ejercicio1(self):
        samples = 100000

        juan_dice_samples = self.dice(samples)
        pedro_dice_samples = self.dice(samples)

        juan_winning_with5 = juan_dice_samples[(juan_dice_samples > pedro_dice_samples) & (juan_dice_samples == 5)]
        juan_winning = juan_dice_samples[(juan_dice_samples > pedro_dice_samples)]

        print(juan_winning_with5.shape[0] / juan_winning.shape[0])

    def test_ejercicio2(self):
        L = 1
        n_samples = 10000
        y = np.random.uniform(0, L, n_samples)
        x = np.random.uniform(0, L - y)

        print(np.mean(x))
        print(np.std(x) ** 2)

    def test_ejercicio4(self):
        _10_samples = 10
        _1000_samples = 1000
        y_10 = 2 + np.random.normal(0, 1, _10_samples)
        y_1000 = 2 + np.random.normal(0, 1, _1000_samples)

        mean_10 = np.mean(y_10)
        mean_1000 = np.mean(y_1000)
        print("Media (10 muestras) ", mean_10)
        print("Media (1000 muestras) ", mean_1000)

        sn_10 = np.sum((y_10 - mean_10) * +2) / _10_samples
        sn_10_unbiased = np.sum((y_10 - mean_10) * +2) / (_10_samples - 1)
        sn_1000 = np.sum((y_1000 - mean_1000) ** 2) / _1000_samples
        sn_1000_unbiased = np.sum((y_1000 - mean_1000) ** 2) / (_1000_samples - 1)

        print("Desvio sesgado (10 muestras) ", sn_10)
        print("Desvio no sesgado (10 muestras) ", sn_10_unbiased)
        print("Desvio sesgado (1000 muestras) ", sn_1000)
        print("Desvio no sesgado (1000 muestras) ", sn_1000_unbiased)

        if __name__ == '__main__':
            unittest.main()
