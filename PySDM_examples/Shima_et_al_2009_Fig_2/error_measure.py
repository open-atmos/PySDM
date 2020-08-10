"""
Created at 10.08.2020
"""

import numpy as np


def error_measure(y, y_true, x):
    errors = y_true - y
    errors = errors[0:-1] + errors[1:]
    dx = x[1:] - x[:-1]
    errors *= dx
    errors[0] /= 2
    errors[1] /= 2
    error = np.sum(np.abs(errors))
    return error