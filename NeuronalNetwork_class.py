import numpy as np


class LayerPerceptron:

    # Constructor Init weights and biases
    def __init__(self, n_inputs, n_hidden, n_output, learning_rate, activation):

        self.activation = activation
        self.learning_rate = learning_rate
        self.bias_h = np.zeros((n_hidden, 1))
        self.bias_o = np.zeros((n_output, 1))

        # Init the weights using
        if self.activation == 'Sigmoid':

            r = (6 / (n_inputs + n_output)) ** 0.5
            self.weight_ih = np.random.uniform(-r, r, (n_inputs, n_hidden))
            self.weight_ho = np.random.uniform(-r, r, (n_hidden, n_output))

        elif self.activation == 'Step':

            self.weight_ih = np.random.uniform(-1, 1, (n_inputs, n_hidden))
            self.weight_ho = np.random.uniform(-1, 1, (n_hidden, n_output))

        elif self.activation == 'ReLU':

            r = 1.42 * ((6 / (n_inputs + n_output)) ** 0.5)
            self.weight_ih = np.random.uniform(-r, r, (n_inputs, n_hidden))
            self.weight_ho = np.random.uniform(-r, r, (n_hidden, n_output))

    # Train the data
    def fit_forward(self, inputs, label, test_mode):
        self.inputs = np.transpose(inputs)
        self.label = label
        self.output_h = np.dot(np.transpose(self.weight_ih), self.inputs)
        # print("these are hidden layer inputs")
        # print(self.output_h)

        # layer input to hidden

        if self.activation == 'ReLU':
            stock = []
            for out in self.output_h:
                if out > 0:
                    stock.append(out)
                elif out <= 0:
                    stock.append(0.01 * out)
            self.output = np.array(stock)

        elif self.activation == 'Sigmoid':
            stock = []
            for out in self.output_h: stock.append([(1 / (1 + np.exp(-out)))])
            self.output = np.array(stock)
            # print(self.output)

        elif self.activation == 'Step':
            stock = []
            for out in self.output_h:
                if out > 0:
                    stock.append([1])
                elif out <= 0:
                    stock.append([0])
            self.output = np.array(stock)

            # print(self.output)

        else:
            print("Wrong input for Activation function")

        ## layer hidden to output

        self.output_o = np.dot(np.transpose(self.weight_ho), self.output)
        # print("these are output layer inputs")
        # print(self.output_o)

        if self.activation == 'ReLU':
            stock = []
            for out in self.output_o:
                if out > 0:
                    stock.append(out)
                elif out <= 0:
                    stock.append([0])
            self.output_final = np.array(stock)

        elif self.activation == 'Sigmoid':
            stock = []
            for out in self.output_o: stock.append([(1 / (1 + np.exp(-out[0])))])
            self.output_final = np.array(stock)

        elif self.activation == 'Step':
            stock = []
            for out in self.output_o:
                if out > 0:
                    stock.append([1])
                elif out <= 0:
                    stock.append([0])
            self.output_final = np.array(stock)

            # print(self.output_final)

        # Backpropagation

        if test_mode == 'test':
            pass
        elif test_mode == 'train':

            if self.activation == 'Sigmoid':

                self.delta_Sigmoid_o = self.output_final * (np.ones(len(self.output_final)) - self.output_final)
                self.delta_error_o = ((self.output_final - label) ** 2) ** 0.5

                self.delta_error_h = np.dot(self.weight_ho, self.delta_error_o)
                self.delta_Sigmoid_h = self.output * (np.ones(len(self.output)) - self.output)

                # print(self.weight_ho)
                # print('')
                # print(self.weight_ih)

                self.weight_ho += self.learning_rate * np.dot((self.delta_Sigmoid_o * self.delta_error_o), self.output_o)
                self.weight_ih += self.learning_rate * np.dot((self.delta_Sigmoid_h * self.delta_error_h), self.output_h)

                # self.bias_h -= self.delta_error_h * self.learning_rate
                # self.bias_o -= self.delta_error_o * self.learning_rate

            elif self.activation == 'Step':

                self.delta_error_o = ((self.output_final - label) ** 2) ** 0.5
                self.delta_error_h = np.dot(self.weight_ho, self.delta_error_o)

                # print(self.weight_ho)
                # print('')
                # print(self.weight_ih)

                self.weight_ho += self.delta_error_o * self.learning_rate
                self.weight_ih += self.delta_error_h * self.learning_rate

                # self.bias_h += self.delta_error_h * (self.learning_rate ** 3)
                # self.bias_o += self.delta_error_o * (self.learning_rate ** 3)

            elif self.activation == 'ReLU':

                self.delta_error_o = ((self.output_final - label) ** 2) ** 0.5
                self.delta_error_h = np.dot(self.weight_ho, self.delta_error_o)

                self.weight_ho += self.delta_error_o * self.learning_rate
                self.weight_ih += self.delta_error_h * self.learning_rate