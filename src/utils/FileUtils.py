import numpy as np
import pickle


class FileUtils:
    @staticmethod
    def save_dataset(dataset: np.ndarray, path: str) -> bool:
        file = None
        try:
            file = open(path, "wb")
            pickle.dump(dataset, file)
            return True
        except Exception:
            return False
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


