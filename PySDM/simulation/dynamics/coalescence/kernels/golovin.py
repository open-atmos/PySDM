"""
Created at 07.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import mpmath
import numpy as np


class Golovin:
    def __init__(self, b, x='volume'):
        self.b = b
        self.x = x

    def __call__(self, particles, output, is_first_in_pair):
        particles.sum_pair(output, self.x, is_first_in_pair)
        particles.backend.multiply(output, self.b)

    def analytic_solution(self, x, t, x_0, N_0):
        tau = 1 - mpmath.exp(-N_0 * self.b * x_0 * t)

        if isinstance(x, np.ndarray):
            func = np.vectorize(lambda i: Golovin.analytic_solution_helper(x[int(i)], tau, x_0))
            result = np.fromfunction(func, x.shape, dtype=float)
            return result

        return Golovin.analytic_solution_helper(x, tau, x_0)

    @staticmethod
    def analytic_solution_helper(x, tau, x_0):
        result = float(
            (1 - tau) *
            1 / (x * mpmath.sqrt(tau)) *
            mpmath.besseli(1, 2 * x / x_0 * mpmath.sqrt(tau)) *
            mpmath.exp(-(1 + tau) * x / x_0)
        )
        return result
