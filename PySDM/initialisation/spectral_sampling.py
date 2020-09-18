"""
Crated at 2019
"""

import numpy as np
from typing import Tuple


default_cdf_range = (.01, .99)


class SpectralSampling:
    def __init__(self, spectrum, radius_range:[None, Tuple[float, float]]):
        self.spectrum = spectrum

        if radius_range is None:
            self.radius_range = spectrum.percentiles(default_cdf_range)
        else:
            assert len(radius_range) == 2
            assert radius_range[0] > 0
            assert radius_range[1] > radius_range[0]
            self.radius_range = radius_range

    @staticmethod
    def _sample(grid, spectrum):
        x = grid[1: -1: 2]
        cdf = spectrum.cumulative(grid[0::2])
        y_float = cdf[1:] - cdf[0:-1]

        return x, y_float


class Linear(SpectralSampling):
    def __init__(self, spectrum, radius_range: [None, Tuple[float, float]]):
        super().__init__(spectrum, radius_range)

    def sample(self, n_sd):
        grid = np.linspace(*self.radius_range, num=2 * n_sd + 1)
        return self._sample(grid, self.spectrum)


class Logarithmic(SpectralSampling):
    def __init__(self, spectrum, radius_range: [None, Tuple[float, float]]):
        super().__init__(spectrum, radius_range)
        self.start = np.log10(radius_range[0])
        self.stop = np.log10(radius_range[1])

    def sample(self, n_sd):
        grid = np.logspace(self.start, self.stop, num=2 * n_sd + 1)
        return self._sample(grid, self.spectrum)


class ConstantMultiplicity(SpectralSampling):
    def __init__(self, spectrum, radius_range):
        super().__init__(spectrum, radius_range)

        self.cdf_range = (
            spectrum.cumulative(self.radius_range[0]),
            spectrum.cumulative(self.radius_range[1])
        )
        assert 0 < self.cdf_range[0] < 1
        assert self.cdf_range[0] < self.cdf_range[1] < 1

        self.spectrum = spectrum

    def sample(self, n_sd):
        cdf_arg = np.linspace(self.cdf_range[0], self.cdf_range[1], num=2 * n_sd + 1)
        cdf_arg /= self.spectrum.norm_factor
        percentiles = self.spectrum.percentiles(cdf_arg)

        assert np.isfinite(percentiles).all()

        return self._sample(percentiles, self.spectrum)
