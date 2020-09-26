import unittest

from src.deepLearning.activationFunctions.SigmoidActivationFunction import SigmoidActivationFunction
from src.machineLearning.metrics.PercentageError import PercentageError
from src.machineLearning.metrics.Precision import Precision
from src.machineLearning.metrics.Recall import Recall
from src.machineLearning.metrics.Selectivity import Selectivity
from src.utils.DatasetUtils import DatasetUtils
from src.deepLearning.NeuralNetwork import NeuralNetwork
from src.deepLearning.Layer import Layer
import matplotlib.pyplot as plt


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

        train_error, validation_error = nn.fit(x_train, y_train, n_epochs=5000, learning_rate=0.005, batch_size=16)

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
        plt.show()


if __name__ == '__main__':
    unittest.main()
