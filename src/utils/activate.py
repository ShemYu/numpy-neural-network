import numpy as np


def threshold_function(x):
    y = x > 0
    return y.astype(int)


def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))


def tangent_function(x):
    return np.tanh(x)


def relu_function(x):
    return np.maximum(0, x)


def softmax_function(x):
    return np.exp(x) / np.sum(np.exp(x))
