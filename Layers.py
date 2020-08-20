import numpy as np
from NeuronalNetwork_class import LayerPerceptron
from DataGeneration import Point, data

inputs, label = data()
# print(inputs)
# print(label)

layer = LayerPerceptron(2, 2, 1, 0.15, 'Step')

for i in range(0, 200):

        layer.fit_forward(inputs[i], label[i], 'train')
        next_input_1 = layer.output_final
        print(next_input_1)
        print('')
