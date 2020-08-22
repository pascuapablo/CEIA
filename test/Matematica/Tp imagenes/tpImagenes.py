import os
import unittest

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_something(self):
        file = 'tango.jpg'
        img_orig = mpimg.imread(file)

        A = np.mean(img_orig, -1)
        img = plt.imshow(A)
        img.set_cmap('gray')


        U, S, VT = np.linalg.svd(A)  # Se calcula la SVD de la matriz con la que estamos trabajando.,

        for porcentaje_de_informacion in range(20, 100, 10):
            k = 0
            for i in range(len(S)):
                if sum(S[0:i]) / sum(S) > porcentaje_de_informacion / 100:
                    k = i - 1
                    break

            Sigma = np.diag(S)

            imagen_reducida = U[:, :k] @ Sigma[0:k, :k] @ VT[:k, :]
            plt.figure()

            img2 = plt.imshow(imagen_reducida)
            img2.set_cmap('gray')
            plt.title("informacion = " + str(porcentaje_de_informacion))
            file_red = os.path.splitext(file)[0] + '_red_' + str(k) + os.path.splitext(file)[1]
            mpimg.imsave(file_red, imagen_reducida, cmap="gray")
            sizeOrig = os.stat(file).st_size
            sizeRed = os.stat(file_red).st_size
            reduc = 100 * (sizeOrig - sizeRed) / sizeOrig  # Porcentaje de reducción.,
            print(
                f'Si me quedo con el {porcentaje_de_informacion}% de informacion el archivo pesa  {reduc:.2f} % menos.'
                f' Me quedé con las primeras {k} componentes')
        plt.show()

if __name__ == '__main__':
    unittest.main()
