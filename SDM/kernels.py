"""
Created at 07.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
import scipy.special as ss


class Golovin:
    def __init__(self, b):
        self.b = b

    def __call__(self, m1, m2):
        result = self.b * (m1 + m2)
        return result

    def analytic_solution(self, x, t, x_0, N_0):
        tau = 1 - np.exp(-N_0 * self.b * x_0 * t)
        result = (1 - tau) * \
                 1 / (x * np.sqrt(tau)) * \
                 ss.iv(1, 2 * x / x_0 * np.sqrt(tau)) * \
                 np.exp(-(1 + tau) * x / x_0)
        return result
