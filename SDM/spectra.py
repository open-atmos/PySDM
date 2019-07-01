"""
Created at 03.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from scipy.stats import lognorm
from scipy.stats import expon
import math
import numpy as np


class Spectrum:
    def __init__(self, distribution, distribution_params, n_part):
        self.distribution_params = distribution_params  # (loc, scale)
        self.n_part = n_part
        self.distribution = distribution

    def size_distribution(self, m):
        result = self.n_part * self.distribution.pdf(m, *self.distribution_params)
        return result

    def stats(self, moments):
        result = self.distribution.stats(*self.distribution_params, moments)
        return result

    def cumulative(self, m):
        result = self.n_part * self.distribution.cdf(m, *self.distribution_params)
        return result

    def percentiles(self, cdfarg):
        result = self.distribution.ppf(cdfarg, *self.distribution_params)
        return result


class Exponential(Spectrum):
    def __init__(self, n_part, m_mode, s_geom):  # TODO change name of params?
        super().__init__(expon, (m_mode, s_geom), n_part)


class Lognormal(Spectrum):
    def __init__(self, n_part, m_mode, s_geom):
        super().__init__(lognorm, (math.log(s_geom), 0, m_mode), n_part)
