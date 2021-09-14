"""
Classes representing particle size spectra (based on SciPy.stats logic)
"""
from scipy.stats import lognorm
from scipy.stats import expon
import math
import numpy as np
from scipy.interpolate import interp1d
from PySDM.initialisation.spectral_sampling import default_cdf_range

default_interpolation_grid = tuple(np.linspace(*default_cdf_range, 999))


class Spectrum:

    def __init__(self, distribution, distribution_params, norm_factor):
        self.distribution_params = distribution_params  # (loc, scale)
        self.norm_factor = norm_factor
        self.distribution = distribution

    def size_distribution(self, x):
        result = self.norm_factor * self.distribution.pdf(x, *self.distribution_params)
        return result

    def stats(self, moments):
        result = self.distribution.stats(*self.distribution_params, moments)
        return result

    def cumulative(self, x):
        result = self.norm_factor * self.distribution.cdf(x, *self.distribution_params)
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

class TopHat:
    def __init__(self, norm_factor, endpoints):
        self.norm_factor = norm_factor
        self.endpoints = endpoints
        self._mn = endpoints[0]
        self._mx = endpoints[1]

    def cumulative(self, x):
        return self.norm_factor * np.minimum(1, np.maximum(0, (x - self._mn) / (self._mx - self._mn)))

    def percentiles(self, cdf_values):
        return (self._mx - self._mn) * (np.asarray(cdf_values) + self._mn / (self._mx - self._mn))

class Sum:

    def __init__(self, spectra: tuple, interpolation_grid=default_interpolation_grid):
        self.spectra = spectra
        self.norm_factor = sum((s.norm_factor for s in self.spectra))
        p = [s.percentiles(interpolation_grid) for s in self.spectra]
        x = np.zeros(len(interpolation_grid) * len(self.spectra) + 1)
        x[1:] = np.concatenate(p)
        y = self.cumulative(x) / self.norm_factor
        self.inverse_cdf = interp1d(y, x)

    def size_distribution(self, x):
        result = 0.
        for spectrum in self.spectra:
            result += spectrum.size_distribution(x)
        return result

    def cumulative(self, x):
        result = 0.
        for spectrum in self.spectra:
            result += spectrum.cumulative(x)
        return result

    def percentiles(self, cdf_values):
        return self.inverse_cdf(cdf_values)
