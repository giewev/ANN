import numpy as np
import math
import random

def vector_sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class Network(object):
    def __init__(self, inputs, hiddens, outputs):
        self.learning_rate = 0.1
        self.momentumFactor = 0.9
        self.M_ij = np.zeros((inputs + 1, hiddens))
        self.M_jk = np.zeros((hiddens + 1, outputs))
        self.W_ij = np.random.rand(inputs + 1, hiddens) - 0.5
        self.W_jk = np.random.rand(hiddens + 1, outputs) - 0.5
        self.activation = vector_sigmoid
        self.activation_derivative = sigmoid_derivative

    def calculate(self, inputs):
        input_biased = np.append(inputs, [[1]], axis = 1)
        hiddens = self.activation(np.dot(input_biased, self.W_ij))
        hiddens_biased = np.append(hiddens, [[1]], axis = 1)
        return self.activation(np.dot(hiddens_biased, self.W_jk))

    def train(self, inputs, targets):
        input_biased = np.append(inputs, [[1]], axis = 1)
        input_weighted = np.dot(input_biased, self.W_ij)

        hidden_signal = self.activation(np.dot(input_biased, self.W_ij))
        hidden_biased = np.append(hidden_signal, [[1]], axis = 1)
        hidden_weighted = np.dot(hidden_biased, self.W_jk)

        output_signal = self.activation(hidden_weighted)

        output_error = self.activation_derivative(output_signal) * (targets - output_signal)
        dW_jk = np.dot(hidden_biased.T, output_error) * self.learning_rate

        hidden_error = output_error.sum() * self.activation_derivative(hidden_signal)
        dW_ij = np.dot(input_biased.T, hidden_error) * self.learning_rate

        dW_ij += self.M_ij
        dW_jk += self.M_jk
        self.M_ij = self.momentumFactor * dW_ij
        self.M_jk = self.momentumFactor * dW_jk
        
        self.W_ij += dW_ij
        self.W_jk += dW_jk

xor_samples = []
xor_samples.append((np.array([[0,0]]), np.array([[0]])))
xor_samples.append((np.array([[0,1]]), np.array([[1]])))
xor_samples.append((np.array([[1,0]]), np.array([[1]])))
xor_samples.append((np.array([[1,1]]), np.array([[0]])))

xor_network = Network(2, 2, 1)
for x in range(10000):
    random.shuffle(xor_samples)
    for sample in xor_samples:
        xor_network.train(sample[0], sample[1])

for x in xor_samples:
    print(x[0], xor_network.calculate(x[0]))
