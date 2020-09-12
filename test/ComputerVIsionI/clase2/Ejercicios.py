import unittest
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from src.computerVision.Filters.PixelWise.ChromaticCoordinatesFilter import ChromaticCoordinatesIFilter
from src.computerVision.Filters.PixelWise.WhitePatchFilter import WhitePatchIFilter


class Ejercicios(unittest.TestCase):
    def test_ejercicio_1(self):
        img: np.ndarray = cv.imread('./resources/CoordCrom_1.png')
        img2: np.ndarray = cv.imread('./resources/CoordCrom_2.png')
        img3: np.ndarray = cv.imread('./resources/CoordCrom_3.png')

        filter = ChromaticCoordinatesIFilter()

        img1_codenadas_crom = filter.apply(img)
        img2_codenadas_crom = filter.apply(img2)
        img3_codenadas_crom = filter.apply(img3)

        plt.subplot(231).imshow(img, vmin=0, vmax=255)
        plt.subplot(232).imshow(img2, vmin=0, vmax=255)
        plt.subplot(233).imshow(img3, vmin=0, vmax=255)
        plt.subplot(234).imshow(img1_codenadas_crom, vmin=0, vmax=255)
        plt.subplot(235).imshow(img2_codenadas_crom, vmin=0, vmax=255)
        plt.subplot(236).imshow(img3_codenadas_crom, vmin=0, vmax=255)

        plt.show()
        self.assertEqual(True, False)

    def test_ejercicio_2(self):
        img: np.ndarray = cv.imread('./resources/WP_R.png')
        img2: np.ndarray = cv.imread('./resources/WP_O.png')
        img3: np.ndarray = cv.imread('./resources/WP_B.png')

        filter = WhitePatchIFilter()

        img1_codenadas_crom = filter.apply(img)
        img2_codenadas_crom = filter.apply(img2)
        img3_codenadas_crom = filter.apply(img3)

        plt.subplot(231).imshow(img, vmin=0, vmax=255)
        plt.subplot(232).imshow(img2, vmin=0, vmax=255)
        plt.subplot(233).imshow(img3, vmin=0, vmax=255)
        plt.subplot(234).imshow(img1_codenadas_crom, vmin=0, vmax=255)
        plt.subplot(235).imshow(img2_codenadas_crom, vmin=0, vmax=255)
        plt.subplot(236).imshow(img3_codenadas_crom, vmin=0, vmax=255)

        plt.show()
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
