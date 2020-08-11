import numpy as np

class Perceptron:

    def __init__(self, activationFunction, learning_rate):
        self.weights = np.random.uniform(low=-1, high=1, size=2)
        self.learning_rate = learning_rate
        self.activationFunction = activationFunction

    def guess(self, inputs):
        sum = np.sum(inputs * self.weights)
        return self.activationFunction(sum)

    def train(self, inputs, target):
        guess = self.guess(inputs)
        error = target - guess

        for i in range(0, len(inputs)):
            self.weights[i] += inputs[i] * error * self.learning_rate