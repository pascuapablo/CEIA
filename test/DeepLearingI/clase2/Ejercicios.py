import unittest

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

from src.deepLearning.activationFunctions.SigmoidActivationFunction import SigmoidActivationFunction
from src.machineLearning.metrics.Precision import Precision
from src.machineLearning.metrics.Recall import Recall
from src.machineLearning.metrics.Selectivity import Selectivity
from src.utils.DatasetUtils import DatasetUtils
from src.deepLearning.NeuralNetwork import NeuralNetwork
from src.deepLearning.Layer import Layer
import matplotlib.pyplot as plt

from src.utils.PlotUtils import PlotUtils


class Ejercicios(unittest.TestCase):
    def test_neuralNetwork(self):
        dataset = DatasetUtils(path='./resources/train_data.csv')
        train, test = dataset.split(80)

        y_train = train[:, -1]
        x_train = train[:, :-1]

        y_test = test[:, -1][:, None]
        x_test = test[:, :-1]

        nn = NeuralNetwork(layers=[
            Layer(input_size=2, neurons=3, activation_function=SigmoidActivationFunction()),
            Layer(input_size=3, neurons=2, activation_function=SigmoidActivationFunction()),
            Layer(input_size=2, neurons=1, activation_function=SigmoidActivationFunction())
        ])

        train_error, validation_error = nn.fit(x_train, y_train, n_epochs=3000, learning_rate=0.005, batch_size=16)

        y_predict = nn.predict(x_test)

        metrics = [Recall(), Selectivity(), Precision()]
        for metric in metrics:
            print(metric.get_name() + ":", metric(y_test, y_predict))
        # Recall: 95.74468085106383
        # Selectivity: 98.83720930232558
        # Precision: 94.73684210526315

        plt.figure()
        plt.plot(train_error, c='red', label='train')
        plt.plot(validation_error, c='blue', label='validation')
        plt.legend()
        # La grafica esta guradada en ./resources/trainning_error.png
        plt.figure()

        index_1 = (y_test == 1).reshape(len(y_test))
        index_0 = (y_test == 0).reshape(len(y_test))
        plt.scatter(x_test[index_1, 0], x_test[index_1, 1], c='red')
        plt.scatter(x_test[index_0, 0], x_test[index_0, 1], c='blue')

        PlotUtils().plotDecisionBoundry(x_test, nn, plt)

        plt.show()

        if __name__ == '__main__':
            unittest.main()

    def test_with_keras(self):
        # Comentar para usar GPU
        tf.config.experimental.set_visible_devices([], 'GPU')
        dataset = DatasetUtils(path='./resources/train_data.csv')
        train, test = dataset.split(80)

        y_train = train[:, -1]
        x_train = train[:, :-1]

        y_test = test[:, -1][:, None]
        x_test = test[:, :-1]

        model = Sequential()

        model.add(Dense(128, input_dim=x_train.shape[1]))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation="sigmoid"))

        metrics = [keras.metrics.Precision(name="precision"),
                   keras.metrics.Recall(name="recall"),
                   keras.metrics.BinaryAccuracy(name="Binary Accuracy")]

        model.compile(optimizer=keras.optimizers.Adam(1e-4),
                      loss="binary_crossentropy",
                      metrics=metrics)
        history = model.fit(
            x_train,
            y_train,
            batch_size=16,
            epochs=200,
            verbose=1,
            validation_data=(x_test, y_test)
        )

        PlotUtils().plotHistory(history=history, metrics=metrics, plt=plt)

        plt.figure()
        index_1 = (y_test == 1).reshape(len(y_test))
        index_0 = (y_test == 0).reshape(len(y_test))
        plt.scatter(x_test[index_1, 0], x_test[index_1, 1], c='red')
        plt.scatter(x_test[index_0, 0], x_test[index_0, 1], c='blue')

        PlotUtils().plotDecisionBoundry(x_test, model, plt)

        plt.show()
