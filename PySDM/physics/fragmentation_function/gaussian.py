"""
Gaussian PDF
CDF = 1/2(1 + erf(x/sqrt(2)));
"""
import math


class Gaussian:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def frag_size(const, mu, sigma, rand):
        return mu + sigma / 2 * (1 + math.erf(rand / const.sqrt_two))
