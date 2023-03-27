"""
Gaussian PDF
CDF = 1/2(1 + erf((x-mu)/sigma/sqrt(2)));
Approx erf(x/sqrt(2)) ~ tanh(x*pi/2/sqrt(3))
"""
import numpy as np


class Gaussian:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def frag_size(const, mu, sigma, rand):
        return mu + sigma * 2 * np.sqrt(3) / const.PI * np.arctanh(2 * rand - 1)
