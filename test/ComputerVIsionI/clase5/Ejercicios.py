import unittest

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


class Ejercicios(unittest.TestCase):
    def test_something(self):
        img = cv.imread("./resources/cielo.png")
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_out = self.lbp(img)

        img2 = cv.imread("./resources/piedras1.png")
        img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        img_out2 = self.lbp(img2)

        fig, ax = plt.subplots(2, 3)
        ax[0][0].imshow(img, cmap="gray")
        ax[0][1].imshow(img_out, cmap="gray")
        ax[0][2].hist(img_out.ravel(), bins=256)
        ax[1][0].imshow(img2, cmap="gray")
        ax[1][1].imshow(img_out2, cmap="gray")
        ax[1][2].hist(img_out2.ravel(), bins=256)
        plt.show()

    def lbp(self, image: np.ndarray) -> np.ndarray:
        rows = image.shape[0]
        columns = image.shape[1]
        lbp_image = np.zeros(image.shape)

        for i in range(1, rows - 1):
            for j in range(1, columns - 1):
                a_k = [image[i, j + 1] >= image[i, j],
                       image[i - 1, j + 1] >= image[i, j],
                       image[i - 1, j] >= image[i, j],
                       image[i - 1, j - 1] >= image[i, j],
                       image[i, j - 1] >= image[i, j],
                       image[i + 1, j - 1] >= image[i, j],
                       image[i + 1, j] >= image[i, j],
                       image[i + 1, j + 1] >= image[i, j]]

                newval = 0
                for index, a in enumerate(a_k):
                    val = 1 if a else 0
                    newval += val * (2 ** index)
                lbp_image[i, j] = newval
        return lbp_image


if __name__ == '__main__':
    unittest.main()
