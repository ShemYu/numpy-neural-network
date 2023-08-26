import numpy as np


def mean_square_error(y_hat, y):
    return 0.5 * np.sum((y_hat - y) ** 2)


def cross_entropy_err(y_hat, y):
    delta = 1e-8
    return -np.sum(y * np.log(y_hat + delta))
