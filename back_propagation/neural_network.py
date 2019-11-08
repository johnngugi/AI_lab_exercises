import numpy as np


def bind_function(input, smallest, largest):
    return (input - smallest) / (largest - smallest)


def sigmoid(t):
    return 1/(1+np.exp(-t))


def sigmoid_derivative(p):
    return p * (1 - p)


class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.bind_values(x)
        self.weights1 = np.random.rand(self.input.shape[1], 2)
        self.weights2 = np.random.rand(2, 1)
        self.y = y
        self.bind_values(y)
        self.output = np.zeros(y.shape)

    def bind_values(self, values):
        min_value = values[np.unravel_index(
            np.argmin(values, axis=None), values.shape)]
        max_value = values[np.unravel_index(
            np.argmax(values, axis=None), values.shape)]

        new_values = np.array([bind_function(i, min_value, max_value)
                               for i in np.nditer(values)], dtype=np.float64).reshape(values.shape)

        np.put(values, range(values.size), new_values)

    def feed_foward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        return self.layer2

    def back_prop(self):
        d_weights2 = np.dot(
            self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(
            self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, X, y):
        self.output = self.feed_foward()
        self.back_prop()
