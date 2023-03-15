"""
Gaussian PDF
CDF = 1/2(1 + erf(x/sqrt(2)));
"""
import math

import numpy as np


class Gaussian:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def erfinv(X):
        return np.arctanh(2 * X - 1) * 2 * np.sqrt(3) / np.pi
