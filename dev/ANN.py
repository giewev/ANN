import numpy as np
import math
import random
import struct

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
        self.W_ij = scale(np.random.rand(inputs + 1, hiddens), 0, 1, -1 / (inputs ** 0.5), 1 / (inputs ** 0.5))
        self.W_jk = scale(np.random.rand(hiddens + 1, outputs), 0, 1, -1 / (hiddens ** 0.5), 1 / (hiddens ** 0.5))
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
	# arr = np.ndarray((1, len(image)))
	# for x in range(len(image)):
	# 	arr[0][x] = ord(image[x])
	# return arr

	single_dim = np.fromstring(image, dtype = 'ubyte').astype('float16')
	multi_dim = np.expand_dims(single_dim, axis=1)
	return multi_dim.T

test_labels = load_labels("t10k-labels.idx1-ubyte")
train_labels = load_labels("train-labels.idx1-ubyte")

test_images = load_images("t10k-images.idx3-ubyte")
train_images = load_images("train-images.idx3-ubyte")

net = Network(784, 300, 10)

# example_image = scale(string_to_array_flat(train_images[0]), 0, 256, -1, 1)
# example_label = np.zeros((10))
# example_label[ord(train_labels[0])] = 1
# net.train(example_image, example_label)

# import cv2
# for x in zip(train_images, train_labels):
# 	cv2.imshow('image', string_to_array(x[0], 28, 28))
# 	print(ord(x[1]))
# 	cv2.waitKey(0)
# 	cv2.destroyAllWindows()

# correct_count = 0
# error_count = 0
# for x in zip(test_images, test_labels):
# 	input_arr = scale(string_to_array_flat(x[0]), 0, 256, -1, 1)
# 	target = ord(x[1])

# 	output = np.argmax(net.calculate(input_arr))

# 	if target == output:
# 		correct_count += 1
# 	else:
# 		error_count += 1
# print(correct_count, error_count)


for x in zip(train_images, train_labels):
	input_arr = scale(string_to_array_flat(x[0]), 0, 256, -1, 1)
	target_arr = np.zeros((10))
	target_arr[ord(x[1])] = 1

	net.train(input_arr, target_arr)

correct_count = 0
error_count = 0
for x in zip(test_images, test_labels):
	input_arr = scale(string_to_array_flat(x[0]), 0, 256, -1, 1)
	target = ord(x[1])

	output = np.argmax(net.calculate(input_arr))

	if target == output:
		correct_count += 1
	else:
		error_count += 1
print(correct_count, error_count)