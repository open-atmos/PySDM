"""
Gaussian PDF
CDF = 1/2(1 + erf(x/sqrt(2)));
approximate as erf(x) ~ tanh(ax) with a = sqrt(pi)log(2) as in Vedder 1987
"""
import numpy as np


class Gaussian:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def frag_size(const, mu, sigma, rand):
        return mu - sigma / const.sqrt_two / const.sqrt_pi / np.log(2) * np.log(
            (0.5 + rand) / (1.5 - rand)
        )
