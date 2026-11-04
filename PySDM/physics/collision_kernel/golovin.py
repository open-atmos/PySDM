"""
Safranov-Golovin kernel with analytic solution
(see [Safranov 1962](http://mi.mathnet.ru/dan27172),
[Golovin 1963](http://mi.mathnet.ru/dan27630))
"""

import numpy as np
from scipy import special


class Golovin:  # pylint:disable=too-few-public-methods
    def __init__(self, const):
        assert np.isfinite(const.KERNEL_GOLOVIN_b)

    @staticmethod
    def requested_attributes(_):
        return "volume", "", ""

    # pylint:disable=unused-argument
    @staticmethod
    def liquid_liquid(const, attr0_a, attr0_b, attr1_a, attr1_b, attr2_a, attr2_b):
        return const.KERNEL_GOLOVIN_b * (attr0_a + attr0_b)


def analytic_solution(b, x, t, x_0, N_0):

    def analytic_solution_helper(x, tau, x_0):
        sqrt_tau = np.sqrt(tau)
        return float(
            (1 - tau)
            * 1
            / (x * np.sqrt(tau))
            * special.ive(1, 2 * x / x_0 * sqrt_tau)  # pylint: disable=no-member
            * np.exp(-(1 + tau - 2 * sqrt_tau) * x / x_0)
        )

    tau = 1 - np.exp(-N_0 * b * x_0 * t)

    if isinstance(x, np.ndarray):
        func = np.vectorize(lambda i: analytic_solution_helper(x[int(i)], tau, x_0))
        result = np.fromfunction(func, x.shape, dtype=float)
    else:
        result = analytic_solution_helper(x, tau, x_0)

    return result
