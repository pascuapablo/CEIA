import unittest

import matplotlib.pyplot as plt
import numpy as np

from src.filters.KalmanFilter import KalmanFilter
from src.utils.DatasetUtils import DatasetUtils


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.x0 = np.array([[10.7533, 36.6777, -45.1769, 1.1009, -17.0, 35.7418, -5.7247, 3.4268, 5.2774]]).T
        self.P0 = np.array(
            [[100, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 100, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 100, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0.1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0.1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0.1]])
        self.h = 1
        self.A = np.array([[1, 0, 0, self.h, 0, 0, np.power(self.h, 2) / 2, 0, 0],
                           [0, 1, 0, 0, self.h, 0, 0, np.power(self.h, 2) / 2, 0],
                           [0, 0, 1, 0, 0, self.h, 0, 0, np.power(self.h, 2) / 2],
                           [0, 0, 0, 1, 0, 0, self.h, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, self.h, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, self.h],
                           [0, 0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1]
                           ])

        self.B = np.eye(9)
        self.Q = 0.3 * np.eye(9)

    def test_ejercicio1(self):
        C = np.append(np.eye(3), np.zeros((3, 6)), axis=1)
        measurement_noise = 10
        R = np.eye(3) * measurement_noise

        kf = KalmanFilter(self.x0, self.P0, self.A, self.B, C, self.Q, R)

        ds_p = DatasetUtils(path='./resources/posicion.dat', delimiter=None)
        position = ds_p.get_dataset()[:, 1:]
        y = position + np.random.normal(0, measurement_noise, position.shape)

        # y = position + np.random.uniform(0, np.sqrt(np.sqrt(measurement_noise) * 12), position.shape)
        # y = position + np.random.uniform(- np.sqrt(np.sqrt(measurement_noise) * 12 / 4),
        #                                  np.sqrt(np.sqrt(measurement_noise) * 12), position.shape)
        est_position = np.zeros((len(position), 9))
        error = np.zeros((len(position), 3))

        for i in range(len(position)):
            kf.make_prediction()
            est_x = kf.add_measurement(y[i:(i + 1), :].T)

            est_position[i, :] = est_x.T
            error[i, :] = position[i, :] - est_x.T[:, 0:3]

        self.plot_estimation(position, est_position, "Trayectoria real vs estimada con ruido gausiano")
        self.plot_errors(error, "Error sobre los ejes X, Y y Z con medicion con ruido gausiano")
        plt.show()

    def test_ejercicio2_ruido_normal_con_media(self):
        C = np.append(np.eye(3), np.zeros((3, 6)), axis=1)
        measurement_noise = 10
        R = np.eye(3) * measurement_noise

        kf = KalmanFilter(self.x0, self.P0, self.A, self.B, C, self.Q, R)

        ds_p = DatasetUtils(path='./resources/posicion.dat', delimiter=None)
        position = ds_p.get_dataset()[:, 1:]

        y = position + np.random.uniform(0, np.sqrt(np.sqrt(measurement_noise) * 12), position.shape)
        # y = position + np.random.uniform(- np.sqrt(np.sqrt(measurement_noise) * 12 / 4),
        #                                  np.sqrt(np.sqrt(measurement_noise) * 12), position.shape)
        estimated_position = np.zeros((len(position), 9))
        prediction_error = np.zeros((len(position), 3))

        for i in range(len(position)):
            kf.make_prediction()
            est_x = kf.add_measurement(y[i:(i + 1), :].T)

            estimated_position[i, :] = est_x.T
            prediction_error[i, :] = position[i, :] - est_x.T[:, 0:3]

        self.plot_estimation(position, estimated_position, "Trayectoria real vs estimada ruido normal (0,a)")
        self.plot_errors(prediction_error, "Error sobre los ejes X, Y y Z con medicion con ruido ormal (0,a)")
        plt.show()

    def test_ejercicio2_ruido_normal_sin_media(self):
        C = np.append(np.eye(3), np.zeros((3, 6)), axis=1)
        measurement_noise = 10
        R = np.eye(3) * measurement_noise

        kf = KalmanFilter(self.x0, self.P0, self.A, self.B, C, self.Q, R)

        ds_p = DatasetUtils(path='./resources/posicion.dat', delimiter=None)
        position = ds_p.get_dataset()[:, 1:]

        y = position + np.random.uniform(- np.sqrt(np.sqrt(measurement_noise) * 12 / 4),
                                         np.sqrt(np.sqrt(measurement_noise) * 12 / 4), position.shape)

        estimated_position = np.zeros((len(position), 9))
        prediction_error = np.zeros((len(position), 3))

        for i in range(len(position)):
            kf.make_prediction()
            est_x = kf.add_measurement(y[i:(i + 1), :].T)

            estimated_position[i, :] = est_x.T
            prediction_error[i, :] = position[i, :] - est_x.T[:, 0:3]

        self.plot_estimation(position, estimated_position, "Trayectoria real vs estimada ruido normal (-a,a)")
        self.plot_errors(prediction_error, "Error sobre los ejes X, Y y Z con medicion con ruido ormal (-a,a)")
        plt.show()

    def test_ejercicio3(self):
        C = np.append(np.eye(6), np.zeros((6, 3)), axis=1)
        position_measurement_noise = 10
        speed_measurement_noise = 0.2
        R = np.array([[position_measurement_noise, 0, 0, 0, 0, 0],
                      [0, position_measurement_noise, 0, 0, 0, 0],
                      [0, 0, position_measurement_noise, 0, 0, 0],
                      [0, 0, 0, speed_measurement_noise, 0, 0],
                      [0, 0, 0, 0, speed_measurement_noise, 0],
                      [0, 0, 0, 0, 0, speed_measurement_noise]])

        kf = KalmanFilter(self.x0, self.P0, self.A, self.B, C, self.Q, R)

        ds_p = DatasetUtils(path='./resources/posicion.dat', delimiter=None)
        position = ds_p.get_dataset()[:, 1:]
        y = position + np.random.normal(0, position_measurement_noise, position.shape)

        ds_s = DatasetUtils(path='./resources/velocidad.dat', delimiter=None)
        speed = ds_s.get_dataset()[:, 1:]

        y = np.append(y, speed, axis=1)

        estimated_position = np.zeros((len(position), 9))
        prediction_error = np.zeros((len(position), 3))

        for i in range(len(position)):
            kf.make_prediction()
            est_x = kf.add_measurement(y[i:(i + 1), :].T)
            estimated_position[i, :] = est_x.T
            prediction_error[i, :] = position[i, :] - est_x.T[:, 0:3]

        self.plot_estimation(position, estimated_position, "Trayectoria real vs estimada ruido gausiano")
        self.plot_errors(prediction_error, "Error sobre los ejes X, Y y Z con medicion con ruido gausiano")
        plt.show()

    def plot_estimation(self, target, estimation, title):
        ax = plt.axes(projection='3d')
        ax.set_title(title)
        ax.plot(xs=target[:, 0], ys=target[:, 1], zs=target[:, 2], c='blue', label="trayectoria real")
        ax.plot(xs=estimation[:, 0], ys=estimation[:, 1], zs=estimation[:, 2], c='red',
                label="trayectoria estimada")
        ax.legend()

    def plot_errors(self, error, title):
        fig, axs = plt.subplots(3, 1, sharex=True)
        axs[0].set_title(title)
        axs[0].plot(error[:, 0], label="Error sobre eje X", )
        axs[0].plot(range(len(error)), np.mean(error[:, 0]) * np.ones(len(error)),
                    label="Media " + str(np.mean(error[:, 0]).round(2)))
        axs[0].legend()
        axs[1].plot(error[:, 1], label="Error sobre eje Y", )
        axs[1].plot(range(len(error)), np.mean(error[:, 1]) * np.ones(len(error)),
                    label="Media " + str(np.mean(error[:, 1]).round(2)))
        axs[1].legend()
        axs[2].plot(error[:, 2], label="Error sobre eje Z", )
        axs[2].plot(range(len(error)), np.mean(error[:, 2]) * np.ones(len(error)),
                    label="Media " + str(np.mean(error[:, 2]).round(2)))
        axs[2].legend()


if __name__ == '__main__':
    unittest.main()
