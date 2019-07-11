"""
Created at 07.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import mpmath
import numpy as np


class Golovin:
    def __init__(self, b):
        self.b = b

    def __call__(self, m1, m2):
        result = self.b * (m1 + m2)
        return result

    def analytic_solution(self, x, t, x_0, N_0):
        tau = 1 - mpmath.exp(-N_0 * self.b * x_0 * t)

        if isinstance(x, np.ndarray):
            func = np.vectorize(lambda i: Golovin.helper(x[int(i)], tau, x_0))
            result = np.fromfunction(func, x.shape, dtype=float)
            return result

        return Golovin.helper(x, tau, x_0)

    @staticmethod
    def helper(x, tau, x_0):
        result = float(
            (1 - tau) *
            1 / (x * mpmath.sqrt(tau)) *
            mpmath.besseli(1, 2 * x / x_0 * mpmath.sqrt(tau)) *
            mpmath.exp(-(1 + tau) * x / x_0)
        )
        return result
