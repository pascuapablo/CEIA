import unittest

import numpy as np


class MyTestCase(unittest.TestCase):
    def test_something(self):
        A = np.array([[1, 0], [1, 0], [1, 1]])

        print("[test_something] lambda", np.linalg.eig(A.T @ A))

        u, s, v = np.linalg.svd(A)
        print("[test_something] U", u)
        print("[test_something] sigma", np.diag(s))
        print("[test_something] V", v)
        # print("[test_something] A ", u @ np.diag(s) @ v)

        U = np.array([[0.5, 0.5], [0.5, 0.5], [0.7, -0.71]])
        sigma = np.array([[1.84, 0], [0, 0.76]])
        V = np.array([[0.92, 0.38], [0.38, -0.92]])
        print("[test_something] A", U @ sigma @ V)


if __name__ == '__main__':
    unittest.main()
