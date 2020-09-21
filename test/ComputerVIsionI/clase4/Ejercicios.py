import unittest
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from src.computerVision.FilterUtils import FilterUtils
from src.computerVision.Filters.LinearFilters.DoGFilter import DoGFilter
from src.computerVision.Filters.LinearFilters.GaussianBlurFilter import GaussianBlurFilter
from src.computerVision.Filters.LinearFilters.MedianFilter import MedianFilter
from src.computerVision.Filters.LinearFilters.UnsharpMaskFilter import UnsharpMaskFilter
from src.computerVision.Filters.PixelWise.ChromaticCoordinatesFilter import ChromaticCoordinatesFilter
from src.computerVision.Filters.PixelWise.WhitePatchFilter import WhitePatchFilter


class Clase4(unittest.TestCase):
    def test_iris(self):
        img = cv.imread("./resources/eyes.jpg")
        img_out = cv.cvtColor(img.copy(), cv.COLOR_BGR2RGB)

        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        img = FilterUtils(img).apply_all([
            GaussianBlurFilter()
        ])

        circles = cv.HoughCircles(img,
                                  cv.HOUGH_GRADIENT,
                                  1,
                                  minDist=80,
                                  param1=160,
                                  param2=15,
                                  minRadius=20,
                                  maxRadius=30)

        if circles is None:
            raise Exception("No se encontro nigun criculo")

        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv.circle(img_out, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv.circle(img_out, (i[0], i[1]), 2, (0, 0, 255), 3)

        circles = circles[0, :, 0:2]
        center_y: np.ndarray = np.argsort(circles[:, 1])
        for i in range(0, len(center_y), 2):
            index_1 = center_y[i]
            index_2 = center_y[i + 1]
            norma = np.abs(circles[index_1, :].astype(np.int) - circles[index_2, :].astype(np.int))
            print("distancia del par de ojos " + str(int(i / 2)) + ":", norma[0], "pixeles")

        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(img, cmap='gray')
        axs[1].imshow(img_out)
        plt.show()

    def test_pupilas(self):
        img = cv.imread("./resources/eyes.jpg")
        img_out = cv.cvtColor(img.copy(), cv.COLOR_BGR2RGB)

        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        img = FilterUtils(img).apply_all([
            GaussianBlurFilter(),
            MedianFilter()
        ])
        circles = cv.HoughCircles(img,
                                  cv.HOUGH_GRADIENT,
                                  1,
                                  minDist=90,
                                  param1=100,
                                  param2=15,
                                  minRadius=7,
                                  maxRadius=15)

        if circles is None:
            raise Exception("No se encontro nigun criculo")

        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv.circle(img_out, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv.circle(img_out, (i[0], i[1]), 2, (0, 0, 255), 3)

        circles = circles[0, :, 0:2]
        center_y: np.ndarray = np.argsort(circles[:, 1])
        for i in range(0, len(center_y), 2):
            index_1 = center_y[i]
            index_2 = center_y[i + 1]
            norma = np.abs(circles[index_1, :].astype(np.int) - circles[index_2, :].astype(np.int))
            print("distancia del par de ojos " + str(int(i / 2)) + ":", norma[0], "pixeles")

        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(img, cmap='gray')
        axs[1].imshow(img_out)
        plt.show()


if __name__ == '__main__':
    unittest.main()
