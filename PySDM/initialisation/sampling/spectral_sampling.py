"""
spectral sampling logic incl. linear, logarithmic, uniform-random and constant-multiplicity
 sampling classes
"""

from typing import Optional, Tuple

import numpy as np
from scipy import optimize
from scipy.interpolate import interp1d

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


class AlphaSampling(
    DeterministicSpectralSampling
):  # pylint: disable=too-few-public-methods
    """as in [Matsushima et al. 2023](https://doi.org/10.5194/gmd-16-6211-2023)"""

    def __init__(self, spectrum, alpha, size_range=None, dist_0=None, dist_1=None,interp_points=10000,convert_to=None):
        super().__init__(spectrum, size_range)
        self.alpha = alpha
        if dist_0 is None:
            dist_0 = self.spectrum
        if dist_1 is None:

            if convert_to=="radius":
                def dist_1_inv(y):
                    return 4*np.pi/3*(((3/(4*np.pi)*self.size_range[1])**(1/3) - (3/(4*np.pi)*self.size_range[0])**(1/3)) * y + (3/(4*np.pi)*self.size_range[0])**(1/3))**3
            elif convert_to=="log_radius":
                def dist_1_inv(y):
                    radius_0 = (3/(4*np.pi)*self.size_range[0])**(1/3)
                    radius_1 = (3/(4*np.pi)*self.size_range[1])**(1/3)
                    return 4*np.pi/3*(np.exp((np.log(radius_1) - np.log(radius_0)) * y + np.log(radius_0)))**3
            elif convert_to=="log":
                def dist_1_inv(y):
                    return np.exp((np.log(self.size_range[1]) - np.log(self.size_range[0])) * y + np.log(self.size_range[0]))
            else:
                def dist_1_inv(y):
                    return (self.size_range[1] - self.size_range[0]) * y + self.size_range[0]

        else:
            dist_1_inv = dist_1.percentiles
        self.dist_0_cdf = dist_0.cdf
        self.dist_1_inv = dist_1_inv
        self.x_prime = np.linspace(self.size_range[0], self.size_range[1], num=interp_points)

    def sample(
        self, n_sd, *, backend=None,
    ):  # pylint: disable=unused-argument

        sd_cdf = self.dist_0_cdf(self.x_prime)

        x_sd_cdf = (1 - self.alpha) * self.x_prime + self.alpha * self.dist_1_inv(sd_cdf)

        inv_cdf = interp1d(sd_cdf, x_sd_cdf)

        percent_values = self._find_percentiles(n_sd, backend)
        percentiles = inv_cdf(percent_values)

        return self._sample(percentiles, self.spectrum)

    def _find_percentiles(self, n_sd, backend):
        percent_values = np.linspace(
            default_cdf_range[0], default_cdf_range[1], num=2 * n_sd + 1
        )
        return percent_values


class AlphaSamplingPseudoRandom(
    AlphaSampling
):  # pylint: disable=too-few-public-methods
    """Alpha sampling with pseudo-random values within deterministic percentile bins"""

    def _find_percentiles(self, n_sd, backend):
        num_elements = n_sd
        storage = backend.Storage.empty(num_elements, dtype=float)
        backend.Random(seed=backend.formulae.seed, size=num_elements)(storage)
        u01 = storage.to_ndarray()

        percent_values = np.linspace(
            default_cdf_range[0], default_cdf_range[1], num=2 * n_sd + 1
        )

        for i in range(1, len(percent_values) - 1, 2):
            percent_values[i] = percent_values[i - 1] + u01[i // 2] * (
                percent_values[i + 1] - percent_values[i - 1]
            )

        return percent_values


class AlphaSamplingRandom(AlphaSampling):  # pylint: disable=too-few-public-methods
    """Alpha sampling with uniform random percentile bins"""

    def _find_percentiles(self, n_sd, backend):
        num_elements = 2 * n_sd + 1
        storage = backend.Storage.empty(num_elements, dtype=float)
        backend.Random(seed=backend.formulae.seed, size=num_elements)(storage)
        u01 = storage.to_ndarray()

        percent_values = np.sort(
            default_cdf_range[0] + u01 * (default_cdf_range[1] - default_cdf_range[0])
        )
        return percent_values


class ConstantMultiplicity(AlphaSampling):  # pylint: disable=too-few-public-methods
    def __init__(self, spectrum, size_range=None):
        super().__init__(spectrum, 0, size_range)


class Linear(AlphaSampling):  # pylint: disable=too-few-public-methods
    def __init__(self, spectrum, size_range=None):
        super().__init__(spectrum, 1, size_range)
