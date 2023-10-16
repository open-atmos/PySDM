import numpy as np


def error_measure(y, y_true, x):
    errors = y_true - y
    errors = errors[0:-1] + errors[1:]
    dx = np.diff(x)
    errors *= dx
    errors /= 2
    error = np.sum(np.abs(errors))
    return error
