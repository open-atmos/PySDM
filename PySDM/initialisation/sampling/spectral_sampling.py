"""
spectral discretisation logic incl. linear, logarithmic, and constant-multiplicity
 layouts with deterministic, pseudorandom and quasirandom sampling
"""

import abc
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d

default_cdf_range = (0.00001, 0.99999)


class SpectralSampling:
    def __init__(
        self,
        spectrum,
        *,
        size_range: Optional[Tuple[float, float]] = None,
        error_threshold: Optional[float] = None,
    ):
        self.spectrum = spectrum
        self.error_threshold = error_threshold or 0.01

        if size_range is None:
            self.cdf_range = default_cdf_range
            self.size_range = spectrum.percentiles(self.cdf_range)
        else:
            assert len(size_range) == 2
            assert size_range[0] > 0
            assert size_range[1] > size_range[0]
            self.size_range = size_range
            self.cdf_range = (
                spectrum.cdf(size_range[0]),
                spectrum.cdf(size_range[1]),
            )

    @abc.abstractmethod
    def _sample(self, frac_values):
        pass

    def _sample_with_grid(self, grid):
        x = grid[1:-1:2]
        cdf = self.spectrum.cumulative(grid[0::2])
        y_float = cdf[1:] - cdf[0:-1]

        diff = abs(1 - np.sum(y_float) / self.spectrum.norm_factor)
        if diff > self.error_threshold:
            raise ValueError(
                f"{diff * 100:.3g}% error in total real-droplet number due to sampling "
                f"({len(x)} samples)"
            )

        return x, y_float

    def sample_deterministic(
        self, n_sd, *, backend=None
    ):  # pylint: disable=unused-argument
        return self._sample(
            frac_values=np.linspace(
                self.cdf_range[0], self.cdf_range[1], num=2 * n_sd + 1
            )
        )

    def sample_quasirandom(self, n_sd, *, backend):
        num_elements = n_sd
        storage = backend.Storage.empty(num_elements, dtype=float)
        backend.Random(seed=backend.formulae.seed, size=num_elements)(storage)
        u01 = storage.to_ndarray()

        frac_values = np.linspace(
            self.cdf_range[0], self.cdf_range[1], num=2 * n_sd + 1
        )

        for i in range(1, len(frac_values) - 1, 2):
            frac_values[i] = frac_values[i - 1] + u01[i // 2] * (
                frac_values[i + 1] - frac_values[i - 1]
            )

        return self._sample(frac_values=frac_values)

    def sample_pseudorandom(self, n_sd, *, backend):
        num_elements = 2 * n_sd + 1
        storage = backend.Storage.empty(num_elements, dtype=float)
        backend.Random(seed=backend.formulae.seed, size=num_elements)(storage)
        u01 = storage.to_ndarray()

        frac_values = np.sort(
            self.cdf_range[0] + u01 * (self.cdf_range[1] - self.cdf_range[0])
        )
        return self._sample(frac_values=frac_values)


class Logarithmic(SpectralSampling):
    def _sample(self, frac_values):
        grid = np.exp(
            (np.log(self.size_range[1]) - np.log(self.size_range[0])) * frac_values
            + np.log(self.size_range[0])
        )
        return self._sample_with_grid(grid)


class Linear(SpectralSampling):
    def _sample(self, frac_values):
        grid = self.size_range[0] + frac_values * (
            self.size_range[1] - self.size_range[0]
        )
        return self._sample_with_grid(grid)


class ConstantMultiplicity(SpectralSampling):
    def _sample(self, frac_values):
        grid = self.spectrum.percentiles(frac_values)
        assert np.isfinite(grid).all()
        return self._sample_with_grid(grid)


class AlphaSampling(SpectralSampling):
    """as in [Matsushima et al. 2023](https://doi.org/10.5194/gmd-16-6211-2023)"""

    def __init__(
        self,
        spectrum,
        *,
        alpha,
        dist_1_inv,
        interp_points,
        size_range=None,
        error_threshold: Optional[float] = None,
    ):
        super().__init__(
            spectrum, size_range=size_range, error_threshold=error_threshold
        )
        self.alpha = alpha
        self.dist_0_cdf = self.spectrum.cdf
        self.dist_1_inv = dist_1_inv
        self.x_prime = np.linspace(
            self.size_range[0], self.size_range[1], num=interp_points
        )

    def _sample(self, frac_values):
        if self.alpha == 0:
            frac_values = self.spectrum.percentiles(frac_values)
        elif self.alpha == 1:
            frac_values = self.dist_1_inv(frac_values, self.size_range)
        else:
            sd_cdf = self.dist_0_cdf(self.x_prime)
            x_sd_cdf = (1 - self.alpha) * self.x_prime + self.alpha * self.dist_1_inv(
                sd_cdf, self.size_range
            )
            frac_values = interp1d(sd_cdf, x_sd_cdf)(frac_values)
        return self._sample_with_grid(frac_values)
