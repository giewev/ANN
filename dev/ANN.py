import numpy as np
import math
import random
import struct
import time
import sys
from multiprocessing import Pool
import cProfile
import copy

def vector_sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class Network(object):
    def __init__(self, inputs, hiddens, outputs):
        self.learning_rate = 0.1
        self.momentumFactor = 0.1
        self.decay_rate = 0.0001
        self.M_ij = np.zeros((inputs + 1, hiddens))
        self.M_jk = np.zeros((hiddens + 1, outputs))
        self.W_ij = np.random.randn(inputs + 1, hiddens)
        self.W_jk = np.random.randn(hiddens + 1, outputs)
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
    label = np.zeros((10))
    label[ord(string)] = 1
    return label

def child_print(string):
    print(string)
    sys.stdout.flush()

def sum_deltas(deltas):
    ij = deltas[0][0]
    jk = deltas[0][1]
    for x in range(1, len(deltas)):
        ij += deltas[x][0]
        jk += deltas[x][1]
    return (ij, jk)

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

# import cv2
# for x in zip(train_images, train_labels):
#   cv2.imshow('image', string_to_array(x[0], 28, 28))
#   print(ord(x[1]))
#   cv2.waitKey(0)
#   cv2.destroyAllWindows()

def get_delta_picklable(args):
    ij = args[0]
    jk = args[1]
    inputs = args[2]
    outputs = args[3]

    n = Network(1,1,1)
    n.W_ij = ij
    n.W_jk = jk
    return n.get_delta(inputs, outputs)

def pool_deltas(pool, network, inputs, targets):
    ij = network.W_ij
    jk = network.W_jk
    data = []
    for x in zip(inputs, targets):
        data.append([ij, jk, x[0], x[1]])
    return pool.map(get_delta_picklable, data)

def train(h, l, m, d):
    net = Network(784, h, 10)
    net.learning_rate = l
    net.momentumFactor = m
    net.decay_rate = d

    train_data = zip(train_images, train_labels)
    test_data = zip(test_images, test_labels)

    epoch_count = 600000
    sample_size = 1
    epoch_per_test = 10000

    score_history = []
    version_history = []

    for epoch in range(epoch_count + 1):
        deltas = []
        for x in random.sample(train_data, sample_size):
            deltas.append(net.get_delta(x[0], x[1]))
        net.train_delta(sum_deltas(deltas))

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
            print(correct_count, error_count)
            score_history.append(correct_count / float(correct_count + error_count))
            version_history.append(copy.deepcopy(net))

            evaluation = evaluate_score_history(score_history)
            if evaluation == -1:
                print("Network stagnated, ending training")
                return
            elif evaluation != None:
                score_history = score_history[:evaluation+1]
                version_history = version_history[:evaluation+1]
                net = version_history[-1]
                print("Network performance decreased, rolling back. Current score: " + str(score_history[-1]))



if __name__ == '__main__':
    test_labels = [string_to_label(x) for x in load_labels("t10k-labels.idx1-ubyte")]
    train_labels = [string_to_label(x) for x in load_labels("train-labels.idx1-ubyte")]

    test_images = [scale(string_to_array_flat(x), 0, 256, -1, 1) for x in load_images("t10k-images.idx3-ubyte")]
    train_images = [scale(string_to_array_flat(x), 0, 256, -1, 1) for x in load_images("train-images.idx3-ubyte")]

    train(300, 0.1, 0.9, 0)

