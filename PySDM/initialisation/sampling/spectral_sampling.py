"""
spectral sampling logic incl. linear, logarithmic, uniform-random and constant-multiplicity
 sampling classes
"""
from typing import Tuple, Optional
import numpy as np
from scipy import optimize
from PySDM.physics import constants as const

default_cdf_range = (.00001, .99999)


class SpectralSampling:
    def __init__(self,
                 spectrum,
                 size_range: [None, Tuple[float, float]] = None,
                 error_threshold: Optional[float] = None
     ):
        self.spectrum = spectrum
        self.error_threshold = error_threshold or .01

        if size_range is None:
            if hasattr(spectrum, 'percentiles'):
                self.size_range = spectrum.percentiles(default_cdf_range)
            else:
                self.size_range = [np.nan, np.nan]
                for i in (0, 1):
                    result = optimize.root(
                        lambda x, value=default_cdf_range[i]: spectrum.cdf(x) - value,
                        x0=spectrum.median()
                    )
                    assert result.success
                    self.size_range[i] = result.x
        else:
            assert len(size_range) == 2
            assert size_range[0] > 0
            assert size_range[1] > size_range[0]
            self.size_range = size_range

    def _sample(self, grid, spectrum):
        x = grid[1: -1: 2]
        cdf = spectrum.cumulative(grid[0::2])
        y_float = cdf[1:] - cdf[0:-1]

        diff = abs(1 - np.sum(y_float) / spectrum.norm_factor)
        if diff > self.error_threshold:
            raise Exception(f"{100*diff}% error in total real-droplet number due to sampling")

        return x, y_float


class Linear(SpectralSampling):
    def sample(self, n_sd):
        grid = np.linspace(*self.size_range, num=2 * n_sd + 1)
        return self._sample(grid, self.spectrum)


class Logarithmic(SpectralSampling):
    def __init__(self,
                 spectrum,
                 size_range: [None, Tuple[float, float]] = None,
                 error_threshold: Optional[float] = None
                 ):
        super().__init__(spectrum, size_range, error_threshold)
        self.start = np.log10(self.size_range[0])
        self.stop = np.log10(self.size_range[1])

    def sample(self, n_sd):
        grid = np.logspace(self.start, self.stop, num=2 * n_sd + 1)
        return self._sample(grid, self.spectrum)


class ConstantMultiplicity(SpectralSampling):
    def __init__(self, spectrum, size_range=None):
        super().__init__(spectrum, size_range)

        self.cdf_range = (
            spectrum.cumulative(self.size_range[0]),
            spectrum.cumulative(self.size_range[1])
        )
        assert 0 < self.cdf_range[0] < self.cdf_range[1]

    def sample(self, n_sd):
        cdf_arg = np.linspace(self.cdf_range[0], self.cdf_range[1], num=2 * n_sd + 1)
        cdf_arg /= self.spectrum.norm_factor
        percentiles = self.spectrum.percentiles(cdf_arg)

        assert np.isfinite(percentiles).all()

        return self._sample(percentiles, self.spectrum)


class UniformRandom(SpectralSampling):
    def __init__(self, spectrum, size_range=None, seed=const.default_random_seed):
        super().__init__(spectrum, size_range)
        self.rng = np.random.default_rng(seed)

    def sample(self, n_sd):
        pdf_arg = self.rng.uniform(*self.size_range, n_sd)
        dr = abs(self.size_range[1] - self.size_range[0]) / n_sd
        return pdf_arg, dr * self.spectrum.size_distribution(pdf_arg)
