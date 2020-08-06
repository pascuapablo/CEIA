import numpy as np


class KalmanFilter(object):
    def __init__(self, x0: np.ndarray, P0: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray, Q: np.ndarray,
                 R: np.ndarray):
        self.x = x0
        self.P = P0
        self.A = A
        self.B = B
        self.C = C
        self.R = R
        self.Q = Q

    def make_prediction(self) -> np.ndarray:
        self.x = self.A @ self.x

        self.P = self.A @ self.P @ self.A.T + self.B @ self.Q @ self.B.T
        return self.x

    def add_measurement(self, y: np.ndarray) -> np.ndarray:
        k = self.P @ self.C.T @ np.linalg.inv(self.C @ self.P @ self.C.T + self.R)
        self.x = self.x + k @ (y - self.C @ self.x)
        aux = k @ self.C
        self.P = (np.eye(len(aux)) - k @ self.C) @ self.P
        return self.x
