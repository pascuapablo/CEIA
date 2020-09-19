import unittest
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from src.computerVision.FilterUtils import FilterUtils
from src.computerVision.Filters.LinearFilters.BilateralFilter import BilateralFilter
from src.computerVision.Filters.LinearFilters.GaussianBlurFilter import GaussianBlurFilter
from src.computerVision.Filters.LinearFilters.SobelXFilter import SobelXFilter
from src.computerVision.Filters.LinearFilters.SobelYFilter import SobelYFilter
from src.computerVision.Filters.LinearFilters.UnsharpMaskFilter import UnsharpMaskFilter
from src.computerVision.Filters.PixelWise.ChromaticCoordinatesFilter import ChromaticCoordinatesFilter
from src.computerVision.Filters.PixelWise.RemoveZeros import RemoveZeros
from src.computerVision.Filters.PixelWise.WhitePatchFilter import WhitePatchFilter


class Ejercicios(unittest.TestCase):
    def test_calculo_de_gradientes(self):
        img: np.ndarray = cv.imread('./resources/metalgrid.jpg')
        img = np.mean(img, axis=2)
        img_filtered = FilterUtils().applyAll(img, [
            GaussianBlurFilter((5, 5), 2),
            RemoveZeros()])

        Gx = FilterUtils().applyAll(img_filtered, [
            SobelXFilter(),
            RemoveZeros()])
        Gy = FilterUtils().applyAll(img_filtered, [
            SobelYFilter(),
            RemoveZeros()])

        modulo = np.sqrt(Gx ** 2 + Gy ** 2).astype(np.uint8)
        # threshold = 200
        # index = modulo > threshold
        # modulo[index] = 0
        # modulo[index] = 255

        dir = np.arctan(Gy / Gx) * 180 / np.pi
        dir[(dir < 22.5) & (dir > -22.5)] = 0
        dir[(dir > 22.5) & (dir < 67.5)] = -45
        dir[(dir > 67.5) & (dir < 112.5)] = -90
        dir[(dir > 112.5) & (dir < 157.5)] = -135
        dir[(dir > 157.5) | (dir < -157.5)] = 0
        dir[(dir < -22.5) & (dir > -67.5)] = 45
        dir[(dir < -67.5) & (dir > -112.5)] = 90
        dir[(dir < -112.5) & (dir > -157.5)] = 135

        dir.astype(np.uint8)
        print(np.unique(modulo), np.max(modulo))
        #
        # dir[dir == 45] = 100
        # dir[dir == 90] = 200

        print("dir", np.unique(dir))
        fig, axs = plt.subplots(2, 2)

        axs[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
        axs[0, 0].set_title("original")
        axs[0, 1].imshow(img_filtered, cmap='gray', vmin=0, vmax=255)
        axs[0, 1].set_title("filtrada")

        axs[1, 0].imshow(modulo, cmap='gray', vmin=0, vmax=255)
        axs[1, 0].set_title("Modulo")

        axs[1, 1].imshow(dir, cmap='gray', vmin=0, vmax=255)
        axs[1, 1].set_title("direccion")

        plt.show()

    def test_calculo_de_gradientes_tela(self):
        img: np.ndarray = cv.imread('./resources/tela2.jpg')
        img = np.mean(img, axis=2).astype(np.uint8)
        img_filtered = FilterUtils().applyAll(img, [
            BilateralFilter(10),
            WhitePatchFilter(),
            RemoveZeros()])

        Gx = FilterUtils().applyAll(img_filtered, [
            SobelXFilter(),
            RemoveZeros()])
        Gy = FilterUtils().applyAll(img_filtered, [
            SobelYFilter(),
            RemoveZeros()])

        modulo = np.sqrt(Gx ** 2 + Gy ** 2).astype(np.uint8)
        # threshold = 200
        # index = modulo > threshold
        # modulo[index] = 0
        # modulo[index] = 255

        dir = np.arctan(Gy / Gx) * 180 / np.pi
        dir[(dir < 22.5) & (dir > -22.5)] = 0
        dir[(dir > 22.5) & (dir < 67.5)] = -45
        dir[(dir > 67.5) & (dir < 112.5)] = -90
        dir[(dir > 112.5) & (dir < 157.5)] = -135
        dir[(dir > 157.5) | (dir < -157.5)] = 0
        dir[(dir < -22.5) & (dir > -67.5)] = 45
        dir[(dir < -67.5) & (dir > -112.5)] = 90
        dir[(dir < -112.5) & (dir > -157.5)] = 135

        dir.astype(np.uint8)
        print(np.unique(modulo), np.max(modulo))
        #
        # dir[dir == 45] = 100
        # dir[dir == 90] = 200

        print("dir", np.unique(dir))
        fig, axs = plt.subplots(2, 2)

        axs[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
        axs[0, 0].set_title("original")
        axs[0, 1].imshow(img_filtered, cmap='gray', vmin=0, vmax=255)
        axs[0, 1].set_title("filtrada")

        axs[1, 0].imshow(modulo, cmap='gray', vmin=0, vmax=np.max(modulo))
        axs[1, 0].set_title("Modulo")

        axs[1, 1].imshow(dir, cmap='gray', vmin=0, vmax=255)
        axs[1, 1].set_title("direccion")

        plt.show()


if __name__ == '__main__':
    unittest.main()
