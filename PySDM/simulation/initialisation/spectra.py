"""
Created at 03.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from scipy.stats import lognorm
from scipy.stats import expon
import math


class Spectrum:
    def __init__(self, distribution, distribution_params, norm_factor):
        self.distribution_params = distribution_params  # (loc, scale)
        self.norm_factor = norm_factor
        self.distribution = distribution

    def size_distribution(self, m):
        result = self.norm_factor * self.distribution.pdf(m, *self.distribution_params)
        return result

    def stats(self, moments):
        result = self.distribution.stats(*self.distribution_params, moments)
        return result

    def cumulative(self, m):
        result = self.norm_factor * self.distribution.cdf(m, *self.distribution_params)
        return result

    def percentiles(self, cdf_arg):
        result = self.distribution.ppf(cdf_arg, *self.distribution_params)
        return result


class Exponential(Spectrum):
    def __init__(self, norm_factor, scale):
        super().__init__(expon, (
            0,     # loc
            scale  # scale = 1/lambda
        ), norm_factor)


class Lognormal(Spectrum):
    def __init__(self, norm_factor: float, m_mode: float, s_geom: float):
        super().__init__(lognorm, (math.log(s_geom), 0, m_mode), norm_factor)


# TODO
class Sum:
    def __init__(self, spectra: tuple):
        self.spectra = spectra

    def size_distribution(self, m):
        result = 0.
        for spectrum in self.spectra:
            result += spectrum.size_distribution(m)
        return result

    def cumulative(self, m):
        result = 0.
        for spectrum in self.spectra:
            result += spectrum.cumulative(m)
        return result

