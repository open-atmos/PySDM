"""
spectral sampling logic incl. linear, logarithmic, uniform-random and constant-multiplicity
 sampling classes
"""

from typing import Optional, Tuple

import numpy as np
from scipy import optimize

default_cdf_range = (0.00001, 0.99999)


class SpectralSampling:  # pylint: disable=too-few-public-methods
    def __init__(self, spectrum, size_range: Optional[Tuple[float, float]] = None):
        self.spectrum = spectrum

        if size_range is None:
            if hasattr(spectrum, "percentiles"):
                self.size_range = spectrum.percentiles(default_cdf_range)
            else:
                self.size_range = [np.nan, np.nan]
                for i in (0, 1):
                    result = optimize.root(
                        lambda x, value=default_cdf_range[i]: spectrum.cdf(x) - value,
                        x0=spectrum.median(),
                    )
                    assert result.success
                    self.size_range[i] = result.x
        else:
            assert len(size_range) == 2
            assert size_range[0] > 0
            assert size_range[1] > size_range[0]
            self.size_range = size_range


class DeterministicSpectralSampling(
    SpectralSampling
):  # pylint: disable=too-few-public-methods
    # TODO #1031 - error_threshold will be also used in non-deterministic sampling
    def __init__(
        self,
        spectrum,
        size_range: Optional[Tuple[float, float]] = None,
        error_threshold: Optional[float] = None,
    ):
        super().__init__(spectrum, size_range)
        self.error_threshold = error_threshold or 0.01

    def _sample(self, grid, spectrum):
        x = grid[1:-1:2]
        cdf = spectrum.cumulative(grid[0::2])
        y_float = cdf[1:] - cdf[0:-1]

        diff = abs(1 - np.sum(y_float) / spectrum.norm_factor)
        if diff > self.error_threshold:
            raise ValueError(
                f"{diff * 100:.3g}% error in total real-droplet number due to sampling "
                f"({len(x)} samples)"
            )

        return x, y_float


class Linear(DeterministicSpectralSampling):  # pylint: disable=too-few-public-methods
    def sample(self, n_sd, *, backend=None):  # pylint: disable=unused-argument
        grid = np.linspace(*self.size_range, num=2 * n_sd + 1)
        return self._sample(grid, self.spectrum)


class Logarithmic(
    DeterministicSpectralSampling
):  # pylint: disable=too-few-public-methods
    def __init__(
        self,
        spectrum,
        size_range: [None, Tuple[float, float]] = None,
        error_threshold: Optional[float] = None,
    ):
        super().__init__(spectrum, size_range, error_threshold)
        self.start = np.log10(self.size_range[0])
        self.stop = np.log10(self.size_range[1])

    def sample(self, n_sd, *, backend=None):  # pylint: disable=unused-argument
        grid = np.logspace(self.start, self.stop, num=2 * n_sd + 1)
        return self._sample(grid, self.spectrum)


class ConstantMultiplicity(
    DeterministicSpectralSampling
):  # pylint: disable=too-few-public-methods
    def __init__(self, spectrum, size_range=None):
        super().__init__(spectrum, size_range)

        self.cdf_range = (
            spectrum.cumulative(self.size_range[0]),
            spectrum.cumulative(self.size_range[1]),
        )
        assert 0 < self.cdf_range[0] < self.cdf_range[1]

    def sample(self, n_sd, *, backend=None):  # pylint: disable=unused-argument
        cdf_arg = np.linspace(self.cdf_range[0], self.cdf_range[1], num=2 * n_sd + 1)
        cdf_arg /= self.spectrum.norm_factor
        percentiles = self.spectrum.percentiles(cdf_arg)

        assert np.isfinite(percentiles).all()

        return self._sample(percentiles, self.spectrum)


class UniformRandom(SpectralSampling):  # pylint: disable=too-few-public-methods
    def sample(self, n_sd, *, backend):
        n_elements = n_sd
        storage = backend.Storage.empty(n_elements, dtype=float)
        backend.Random(seed=backend.formulae.seed, size=n_elements)(storage)
        u01 = storage.to_ndarray()

        pdf_arg = self.size_range[0] + u01 * (self.size_range[1] - self.size_range[0])
        dr = abs(self.size_range[1] - self.size_range[0]) / n_sd
        # TODO #1031 - should also handle error_threshold check
        return pdf_arg, dr * self.spectrum.size_distribution(pdf_arg)
