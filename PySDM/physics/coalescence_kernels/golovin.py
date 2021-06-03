import numpy as np
from scipy import special


class Golovin:

    def __init__(self, b):
        self.b = b
        self.core = None

    def __call__(self, output, is_first_in_pair):
        output.sum(self.core.particles['volume'], is_first_in_pair)
        output *= self.b

    def register(self, builder):
        self.core = builder.core
        builder.request_attribute('volume')

    def analytic_solution(self, x, t, x_0, N_0):
        tau = 1 - np.exp(-N_0 * self.b * x_0 * t)

        if isinstance(x, np.ndarray):
            func = np.vectorize(lambda i: Golovin.analytic_solution_helper(x[int(i)], tau, x_0))
            result = np.fromfunction(func, x.shape, dtype=float)
            return result

        return Golovin.analytic_solution_helper(x, tau, x_0)

    @staticmethod
    def analytic_solution_helper(x, tau, x_0):
        sqrt_tau = np.sqrt(tau)
        result = float(
            (1 - tau) *
            1 / (x * np.sqrt(tau)) *
            special.ive(1, 2 * x / x_0 * sqrt_tau) *
            np.exp(-(1 + tau -2 * sqrt_tau) * x / x_0)
        )
        return result
