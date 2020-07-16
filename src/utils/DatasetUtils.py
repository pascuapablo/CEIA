from typing import Tuple
import os

import numpy as np
import pickle


class DatasetUtils(object):

    def __init__(self, dataset: np.ndarray = None, path: str = None):
        if dataset is not None:
            self.ds = dataset
        else:
            self.ds = np.genfromtxt(path, delimiter=',')[1:, :]

    def get_dataset(self) -> np.ndarray:
        return self.ds

    def substract_mean(self, axis) -> np.ndarray:
        mean = np.mean(self.ds, axis, keepdims=True)
        return self.ds - mean

    def normilize_std(self, axis) -> np.ndarray:
        std = np.std(self.ds, axis, keepdims=True)
        self.ds = self.ds / std
        return self.ds

    def pca(self, n_dim):
        x2 = self.substract_mean(axis=0)
        cov_1 = np.cov(x2.T)
        w, v = np.linalg.eig(cov_1)
        idx = w.argsort()[::-1]
        v = v[:, idx]
        return np.matmul(x2, v[:, :n_dim])

    def split(self, percentage: int) -> Tuple[np.ndarray, np.ndarray]:
        randomized_ds = np.random.permutation(self.ds)
        train_rows = np.round((percentage * randomized_ds.shape[0]) / 100).astype(int)
        ds_train = randomized_ds[0:train_rows, :]
        ds_test = randomized_ds[train_rows:, :]

        return ds_train, ds_test

    @staticmethod
    def save_dataset(dataset, path: str) -> bool:
        file = None
        try:
            file = open(path, "wb")
            pickle.dump(dataset, file)
            return True

        finally:
            if file is not None:
                file.close()

    @staticmethod
    def load_dataset(path: str) -> np.ndarray:
        file = None
        try:
            file = open(path, 'rb')
            return pickle.load(file)
        finally:
            file.close()

    @staticmethod
    def load_fromCSV(path: str):
        return np.genfromtxt(path, delimiter=',')
