import numpy as np

from src.deepLearning.activationFunctions.IActivationFunction import IActivationFunction
from src.deepLearning.activationFunctions.SigmoidActivationFunction import SigmoidActivationFunction


class Layer(object):
    def __init__(self, input_size: int = 1, neurons: int = 1,
                 activation_function: IActivationFunction = SigmoidActivationFunction()):
        self.w = np.random.uniform(-1, 1, (input_size, neurons))
        self.b = np.random.uniform(-1, 1, (1, neurons))
        self.activation_function = activation_function

    def compute(self, x):
        """
        second dimension of x must be Layer.input_size
        x => samples or batch_size x layer.input_size x
        w => input_size x n_neurons
        b => n_neurons x 1

        returns
        z => samples x n_neurons
        a => samples x n_neurons
        """

        z = x @ self.w + self.b
        a = self.activation_function.apply(z)

        return z, a
