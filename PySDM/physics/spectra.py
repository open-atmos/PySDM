"""
Classes representing particle size spectra (based on SciPy.stats logic)
"""
import math
import numpy as np
from scipy.stats import lognorm
from scipy.stats import expon
from scipy.interpolate import interp1d
from ..initialisation.spectral_sampling import default_cdf_range

default_interpolation_grid = tuple(np.linspace(*default_cdf_range, 999))


class Spectrum:

    def __init__(self, distribution, distribution_params, norm_factor):
        self.distribution_params = distribution_params  # (loc, scale)
        self.norm_factor = norm_factor
        self.distribution = distribution

    def size_distribution(self, arg):
        result = self.norm_factor * self.distribution.pdf(arg, *self.distribution_params)
        return result

    def pdf(self, arg):
        return self.size_distribution(arg) / self.norm_factor

    def cdf(self, arg):
        return self.distribution.cdf(arg, *self.distribution_params)

    def stats(self, moments):
        result = self.distribution.stats(*self.distribution_params, moments)
        return result

    def cumulative(self, arg):
        result = self.norm_factor * self.distribution.cdf(arg, *self.distribution_params)
        return result

    def percentiles(self, cdf_values):
        result = self.distribution.ppf(cdf_values, *self.distribution_params)
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

    @property
    def s_geom(self):
        return math.exp(self.distribution_params[0])

    @property
    def m_mode(self):
        return self.distribution_params[2]

    @property
    def median(self):
        return self.m_mode

    @property
    def geometric_mean(self):
        return self.s_geom

    def __str__(self):
        return f"{self.__class__.__name__}:"\
               f" (N={self.norm_factor:.3g},"\
               f" m_mode={self.m_mode:.3g},"\
               f" s_geom={self.s_geom:.3g})"


class TopHat:
    def __init__(self, norm_factor, endpoints):
        self.norm_factor = norm_factor
        self.endpoints = endpoints
        self._mn = endpoints[0]
        self._mx = endpoints[1]

    def cumulative(self, arg):
        cdf = np.minimum(1, np.maximum(0, (arg - self._mn) / (self._mx - self._mn)))
        return self.norm_factor * cdf

    def percentiles(self, cdf_values):
        return (self._mx - self._mn) * (np.asarray(cdf_values) + self._mn / (self._mx - self._mn))


class Sum:

    def __init__(self, spectra: tuple, interpolation_grid=default_interpolation_grid):
        self.spectra = spectra
        self.norm_factor = sum((s.norm_factor for s in self.spectra))
        percentiles = [s.percentiles(interpolation_grid) for s in self.spectra]
        cdf_arg = np.zeros(len(interpolation_grid) * len(self.spectra) + 1)
        cdf_arg[1:] = np.concatenate(percentiles)
        cdf = self.cumulative(cdf_arg) / self.norm_factor
        self.inverse_cdf = interp1d(cdf, cdf_arg)

    def size_distribution(self, arg):
        result = 0.
        for spectrum in self.spectra:
            result += spectrum.size_distribution(arg)
        return result

    def cumulative(self, arg):
        result = 0.
        for spectrum in self.spectra:
            result += spectrum.cumulative(arg)
        return result

    def percentiles(self, cdf_values):
        return self.inverse_cdf(cdf_values)
