from math import exp
from random import random, seed

import numpy as np

# Initialize a network


def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs)]}
                    for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden)]}
                    for i in range(n_outputs)]
    network.append(output_layer)
    return network

# Calculate neuron activation for an input


def activate(weights, inputs):
    activation = 0
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

# Transfer neuron activation


def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output


def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# Calculate the derivative of an neuron output


def transfer_derivative(output):
    return output * (1.0 - output)

# Backpropagate error and store in neurons


def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error


def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)-1):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs


def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = y
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


def bind_function(input, smallest, largest):
    return (input - smallest) / (largest - smallest)


def bind_values(values):
    min_value = values[np.unravel_index(
        np.argmin(values, axis=None), values.shape)]
    max_value = values[np.unravel_index(
        np.argmax(values, axis=None), values.shape)]

    new_values = np.array([bind_function(i, min_value, max_value)
                           for i in np.nditer(values)], dtype=np.float64).reshape(values.shape)

    np.put(values, range(values.size), new_values)


x = np.array([
    [30, 40, 50],
    [40, 50, 20],
    [50, 20, 15],
    [20, 15, 60],
    [15, 60, 70],
    [60, 70, 50]
], dtype=np.float64)

# Expected output
y = np.array([20, 15, 60, 70, 50, 40], dtype=np.float64)

bind_values(x)
bind_values(y)

n_inputs = len(x[0]) - 1
n_outputs = len(y)

seed(1)
network = initialize_network(n_inputs, 12, n_outputs)

train_network(network, x, 0.5, 20, 6)
# for layer in network:
#     print(network)
