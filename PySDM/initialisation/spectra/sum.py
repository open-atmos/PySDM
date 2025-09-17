"""
spectrum defined as a sum of an arbitrary set of
 `PySDM.initialisation.impl.spectrum.Spectrum` instances
"""

import numpy as np
from scipy.interpolate import interp1d

from PySDM.initialisation.sampling.spectral_sampling import default_cdf_range

default_interpolation_grid = tuple(np.linspace(*default_cdf_range, 999))


class Sum:
    def __init__(self, spectra: tuple, interpolation_grid=None):
        interpolation_grid = interpolation_grid or default_interpolation_grid
        self.spectra = spectra
        self.norm_factor = sum((s.norm_factor for s in self.spectra))
        percentiles = [s.percentiles(interpolation_grid) for s in self.spectra]
        cdf_arg = np.zeros(len(interpolation_grid) * len(self.spectra) + 1)
        cdf_arg[1:] = np.concatenate(percentiles)
        cdf = self.cumulative(cdf_arg) / self.norm_factor
        self.inverse_cdf = interp1d(cdf, cdf_arg)

    def size_distribution(self, arg):
        result = 0.0
        for spectrum in self.spectra:
            result += spectrum.size_distribution(arg)
        return result

    def cumulative(self, arg):
        result = 0.0
        for spectrum in self.spectra:
            result += spectrum.cumulative(arg)
        return result

    def percentiles(self, cdf_values):
        return self.inverse_cdf(cdf_values)
