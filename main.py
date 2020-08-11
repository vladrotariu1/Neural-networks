import numpy as np
import matplotlib.pyplot as plt
from Perceptron import Perceptron

class Point:

    def __init__(self):
        self.x = np.random.uniform(low=0, high=100)
        self.y = np.random.uniform(low=0, high=100)

        if self.x > self.y: self.label = 1
        else: self.label = -1


def signFunction(number):
    if number >= 0: return 1
    else: return -1


points = [None] * 100
for i in range(0, 100):
    points[i] = Point()

perceptron = Perceptron(signFunction, 0.1)

for point in points:
    inputs = [point.x, point.y]
    label= point.label
    perceptron.train(inputs, label)

for point in points:
    print("point label: {}\nprediction: {}\n\n".format(point.label, perceptron.guess([point.x, point.y])))
