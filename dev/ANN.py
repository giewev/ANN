import numpy as np
import math
import random
import struct
import time
import sys
import multiprocessing
import cProfile
import copy
from functools import partial
import os
import pickle

def vector_sigmoid(x):
    # return 1.0 / (1 + np.exp(-x))
    return 1.0 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    return x * (1 - x)

def vector_tanh(x):
    return 1.7159 * np.tanh(2 * x / 3)

def tanh_derivative(x):
    return 1.1439 * (1 - ((x ** 2) / 2.9443))

def dropout_array(shape, rate):
    # return np.random.binomial([np.ones(shape)],1-rate)[0] * (1.0/(1-rate))
    return np.random.binomial([np.ones(shape)],1-rate)[0]

class Network(object):
    def __init__(self, inputs, hiddens, outputs):
        self.learning_rate = 0.1
        self.momentumFactor = 0.1
        self.decay_rate = 0.0001
        self.M_ij = np.zeros((inputs + 1, hiddens))
        self.M_jk = np.zeros((hiddens + 1, outputs))
        self.W_ij = np.random.randn(inputs + 1, hiddens) / 2
        self.W_jk = np.random.randn(hiddens + 1, outputs) / 2
        self.activation = vector_sigmoid
        self.activation_derivative = sigmoid_derivative

    def calculate(self, inputs):
        input_biased = np.append(inputs, [[1]], axis = 1)
        hiddens = self.activation(np.dot(input_biased, self.W_ij))
        hiddens_biased = np.append(hiddens, [[1]], axis = 1)
        return self.activation(np.dot(hiddens_biased, self.W_jk))

    def train(self, inputs, targets):
        delta = self.get_delta(inputs, targets)
        self.train_delta(delta)

    def get_delta(self, inputs, targets):
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

        return dW_ij, dW_jk

    def get_error(self, inputs, targets):
        input_biased = np.append(inputs, [[1]], axis = 1)
        input_weighted = np.dot(input_biased, self.W_ij)
        hidden_signal = self.activation(np.dot(input_biased, self.W_ij))

        hidden_biased = np.append(hidden_signal, [[1]], axis = 1)
        hidden_weighted = np.dot(hidden_biased, self.W_jk)
        output_signal = self.activation(hidden_weighted)

        output_error = self.activation_derivative(output_signal) * (targets - output_signal)
        hidden_error = output_error.sum() * self.activation_derivative(hidden_signal)

        return hidden_error, output_error

    def train_error(self, error):
        pass

    def get_delta_wrapped(self, data_set):
        return self.get_delta(data_set[0], data_set[1])

    def train_delta(self, delta):
        dW_ij = delta[0] + self.M_ij
        dW_jk = delta[1] + self.M_jk

        self.M_ij = delta[0] * self.momentumFactor
        self.M_jk = delta[1] * self.momentumFactor
        
        if self.decay_rate != 0:
            self.W_ij += delta[0] - self.W_ij * (self.learning_rate * self.decay_rate)
            self.W_jk += delta[1] - self.W_jk * (self.learning_rate * self.decay_rate)
        else:
            self.W_ij += delta[0]
            self.W_jk += delta[1]

class DeepNetwork(object):
    def __init__(self, layer_sizes):
        self.learning_rate = 0.1
        self.momentum_factor = 0
        self.decay_rate = 0
        self.weights = []
        self.momentums = []
        for x in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.weights.append(np.random.randn(x[0] + 1, x[1]) / (x[0] ** 0.5))
            self.momentums.append(np.zeros((x[0] + 1, x[1]), dtype=np.float64))
        self.activation = vector_tanh
        self.activation_derivative = tanh_derivative

    def calculate(self, inputs):
        outputs = inputs
        for x in self.weights:
            outputs = self.activation(np.dot(np.append(outputs, [[1]], axis = 1), x))
        return outputs

    def get_backprop_deltas(self, inputs, targets):
        signal =  [inputs]
        biased =  []
        for x in self.weights:
            biased.append(np.append(signal[-1], [[1]], axis = 1))
            signal.append(self.activation(np.dot(biased[-1], x)))

        errors = []
        errors.append(self.activation_derivative(signal[-1]) * (targets - signal[-1]))
        for x in reversed(signal[1:-1]):
            errors.append(errors[-1].sum() * self.activation_derivative(x))
        errors = reversed(errors)

        deltas = []
        for x in zip(biased, errors):
            deltas.append(np.dot(x[0].T, x[1]) * self.learning_rate)

        return deltas

    def get_quickprop_deltas(self, inputs, targets):
        signal =  [inputs]
        biased =  []
        for x in self.weights:
            biased.append(np.append(signal[-1], [[1]], axis = 1))
            signal.append(self.activation(np.dot(biased[-1], x)))

        errors = []
        errors.append(self.activation_derivative(signal[-1]) * (targets - signal[-1]))
        for x in reversed(signal[1:-1]):
            errors.append(errors[-1].sum() * self.activation_derivative(x))
        errors = reversed(errors)

    def apply_deltas(self, deltas):
        for x in range(len(deltas)):
            change = deltas[x] + self.momentums[x] - (self.weights[x] * self.decay_rate)
            self.weights[x] += change
            self.momentums[x] = change * self.momentum_factor

def load_labels(filename):
    labels = []
    with open(filename, "rb") as f:
        magic_number = struct.unpack('>i', f.read(4))[0]
        label_count = struct.unpack('>i', f.read(4))[0]

        for x in range(label_count):
            labels.append(f.read(1)[0])

    return labels

def load_images(filename):
    images = []
    with open(filename, "rb") as f:
        magic_number = struct.unpack('>i', f.read(4))[0]
        image_count = struct.unpack('>i', f.read(4))[0]
        row_count = struct.unpack('>i', f.read(4))[0]
        column_count = struct.unpack('>i', f.read(4))[0]

        for x in range(image_count):
            images.append(f.read(row_count * column_count))
    return images

def scale(value, old_min, old_max, new_min, new_max):
    return (((value - old_min) / old_max) * new_max) + new_min

def string_to_array(image, x_dim, y_dim):
    arr = np.ndarray((x_dim, y_dim))
    for x in range(x_dim):
        for y in range(y_dim):
            arr[x, y] = ord(image[x * y_dim + y])
    return arr

def string_to_array_flat(image):
    single_dim = np.fromstring(image, dtype = 'ubyte').astype('float64')
    multi_dim = np.expand_dims(single_dim, axis=1)
    return multi_dim.T

def string_to_label(string):
    label = np.zeros((10)) - 1
    label[ord(string)] = 1
    return label

def child_print(string):
    print(string)
    sys.stdout.flush()

def sum_deltas(deltas):
    total_delta = []
    for x in deltas[0]:
        total_delta.append(np.zeros(x.shape))

    for x in deltas:
        for y in range(len(x)):
            total_delta[y] += x[y]

    return total_delta

def evaluate_score_history(scores):
    if len(scores) < 5:
        return None

    if scores[-1] > scores[-2]:
        return None

    if scores[-1] - min([scores[-3], scores[-4]]) > 0.01:
        return None

    if max([scores[-4], scores[-3], scores[-2]]) - scores[-1] > 0.01:
        return scores.index(max(scores))

    return -1

def train_many(train_data, test_data, h, l, m, d):
    pool_size = multiprocessing.cpu_count() / 2
    tasks = [d] * pool_size
    train_part = partial(train_write, train_data, test_data, h, l, m)

    p = multiprocessing.Pool(pool_size)
    p.map(train_part, tasks)

def train_many_deep(train_data, test_data, layer_sizes, l, m, d):
    pool_size = multiprocessing.cpu_count() / 2
    tasks = [d] * pool_size
    train_part = partial(train_write_deep, train_data, test_data, layer_sizes, l, m)

    p = multiprocessing.Pool(pool_size)
    p.map(train_part, tasks)

def strip_file_name(dir):
    for x in range(len(dir) - 1, -1, -1):
        if dir[x] == '/':
            break
    return dir[:x + 1]

def train_write(train_data, test_data, h, l, m, d):
    while True:
        x = train(train_data, test_data, h, l, m, d)
        append = ""
        while os.path.isfile(os.path.realpath(__file__) + "/" + str(x[1]) + append + ".net"):
            if append == "":
                append = "(2)"
            else:
                append = "(" + str(int(append[1:-1]) + 1) + ")"

        f = open(strip_file_name(os.path.realpath(__file__)) + str(x[1]) + append + ".net", 'wb')
        pickle.dump(x[0], f)
        f.close()

def train_write_deep(train_data, test_data, layer_sizes, l, m, d):
    while True:
        x = train_deep(train_data, test_data, layer_sizes, l, m, d)
        append = ""
        while os.path.isfile(os.path.realpath(__file__) + "/" + str(x[1]) + append + ".net"):
            if append == "":
                append = "(2)"
            else:
                append = "(" + str(int(append[1:-1]) + 1) + ")"

        f = open(strip_file_name(os.path.realpath(__file__)) + str(x[1]) + append + ".net", 'wb')
        pickle.dump(x[0], f)
        f.close()

def train(train_data, test_data, h, l, m, d):
    net = Network(784, h, 10)
    net.learning_rate = l
    net.momentumFactor = m
    net.decay_rate = d

    sample_size = 100
    epoch_per_test = 100

    score_history = []
    version_history = []
    rollback_counter = 0
    epoch = 0

    while True:
        deltas = []
        for x in random.sample(train_data, sample_size):
            deltas.append(net.get_delta(x[0], x[1]))
        net.train_delta(sum_deltas(deltas))

        if epoch % epoch_per_test == 0 and epoch != 0:
            net.learning_rate /= 2.0
            # net.decay_rate /= 2.0

            correct_count = 0
            error_count = 0
            output_counts = [0] * 10
            for x in test_data:
                output = net.calculate(x[0])
                output = np.argmax(output)
                target = np.argmax(x[1])
                output_counts[output] += 1

                if target == output:
                    correct_count += 1
                else:
                    error_count += 1
            child_print((os.getpid(), correct_count, error_count))
            score_history.append(correct_count / float(correct_count + error_count))
            version_history.append(copy.deepcopy(net))

            evaluation = evaluate_score_history(score_history)
            if evaluation == -1:
                child_print("Network stagnated. Ending training")
                return (net, max(score_history))
                continue
            elif evaluation != None:
                score_history = score_history[:evaluation+1]
                version_history = version_history[:evaluation+1]
                net = version_history[-1]
                rollback_counter += 1
                if rollback_counter < 10:
                    child_print("Network performance decreased, rolling back. Current score: " + str(score_history[-1]))
                else:
                    child_print("Network performance repeatedly decreased. Ending Training")
                    return (net, max(score_history))
            else:
                rollback_counter = 0
        epoch += 1

def train_deep(train_data, test_data, layer_sizes, l, m, d):
    net = DeepNetwork(layer_sizes)
    net.learning_rate = l
    net.momentum_factor = m
    net.decay_rate = d

    sample_size = 1
    epoch_per_test = 10000

    score_history = []
    version_history = []
    rollback_counter = 0
    epoch = 0

    while True:
        deltas = []
        for x in random.sample(train_data, sample_size):
            deltas.append(net.get_backprop_deltas(x[0], x[1]))
        net.apply_deltas([x / sample_size for x in sum_deltas(deltas)])

        if epoch % epoch_per_test == 0 and epoch != 0:
            # net.learning_rate /= 2.0
            # net.decay_rate /= 2.0

            correct_count = 0
            error_count = 0
            output_counts = [0] * 10
            for x in test_data:
                output = net.calculate(x[0])
                output = np.argmax(output)
                target = np.argmax(x[1])
                output_counts[output] += 1

                if target == output:
                    correct_count += 1
                else:
                    error_count += 1
            child_print((os.getpid(), correct_count, error_count))
            score_history.append(correct_count / float(correct_count + error_count))
            version_history.append(copy.deepcopy(net))

            evaluation = evaluate_score_history(score_history)
            if evaluation == -1:
                child_print("Network stagnated. Ending training")
                return (net, max(score_history))
                continue
            elif evaluation != None:
                score_history = score_history[:evaluation+1]
                version_history = version_history[:evaluation+1]
                net = version_history[-1]
                rollback_counter += 1
                if rollback_counter < 10:
                    child_print("Network performance decreased, rolling back. Current score: " + str(score_history[-1]))
                else:
                    child_print("Network performance repeatedly decreased. Ending Training")
                    return (net, max(score_history))
            else:
                rollback_counter = 0
        epoch += 1

def average(arrays):
    avg = 0
    for x in arrays:
        avg += sum(sum(x)) / prod(x.shape)
    return avg / len(arrays)

def prod(iterable):
    prod_total = 1
    for x in iterable:
        prod_total *= x
    return prod_total

def variance(arrays):
    avg = average(arrays)
    var = 0
    count = 0
    for x in arrays:
        for y in x:
            for z in y:
                var += (avg - z) ** 2
                count += 1
    return var / count

if __name__ == '__main__':
    test_labels = [string_to_label(x) for x in load_labels("t10k-labels.idx1-ubyte")]
    train_labels = [string_to_label(x) for x in load_labels("train-labels.idx1-ubyte")]

    test_images = [scale(string_to_array_flat(x),  0, 256, -3.25, 3.25) + 2.83 for x in load_images("t10k-images.idx3-ubyte")]
    train_images = [scale(string_to_array_flat(x), 0, 256, -3.25, 3.25) + 2.83 for x in load_images("train-images.idx3-ubyte")]

    # print(variance(train_images))
    # print(average(train_images))

    train_many_deep(list(zip(train_images, train_labels)), list(zip(test_images, test_labels)), (784, 100, 10), 0.01, 0, 0)
    # train_once(list(zip(train_images, train_labels)), 300, 0.1, 0, 0)

