"""
Created at 07.06.2019
"""

import mpmath
import numpy as np


class Golovin:
    def __init__(self, b):
        self.b = b
        self.particles = None

    def __call__(self, output, is_first_in_pair):
        output.sum_pair(self.particles.state['volume'], is_first_in_pair)
        output *= self.b

    def register(self, particles_builder):
        self.particles = particles_builder.particles
        particles_builder.request_attribute('volume')

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
