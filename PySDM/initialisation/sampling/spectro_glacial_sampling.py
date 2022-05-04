"""
two-dimensional sampling for singular immersion freezing: constant-multiplicity
 sampling in the freezing temperature vs. immersed surface area phase space
 realised using [vinecopulib](https://vinecopulib.github.io/pyvinecopulib/)
"""
import numpy as np
import pyvinecopulib as pv

from PySDM.initialisation.sampling.spectral_sampling import default_cdf_range
from PySDM.physics import constants as const

# DIM_SIZE = 0
DIM_TEMP = 0
DIM_SURF = 1
N_DIMS = 2


class SpectroGlacialSampling:
    def __init__(
        self,
        *,
        freezing_temperature_spectrum,
        insoluble_surface_spectrum,
        seed=const.default_random_seed
    ):
        self.insoluble_surface_spectrum = insoluble_surface_spectrum
        self.freezing_temperature_spectrum = freezing_temperature_spectrum

        self.insoluble_surface_range = insoluble_surface_spectrum.percentiles(
            default_cdf_range
        )
        self.temperature_range = freezing_temperature_spectrum.invcdf(
            np.asarray(default_cdf_range), insoluble_surface_spectrum.median
        )
        self.seed = seed

    def sample(self, n_sd):
        copula = False
        if copula:
            simulated = pv.Bicop().simulate(n=n_sd, seeds=[self.seed])
            simulated[:, DIM_TEMP] = self.freezing_temperature_spectrum.invcdf(
                1 - simulated[:, DIM_TEMP], self.insoluble_surface_spectrum.median
            )
            simulated[:, DIM_SURF] = self.insoluble_surface_spectrum.percentiles(
                simulated[:, DIM_SURF]
            )
        else:
            simulated = np.empty((n_sd, N_DIMS))
            simulated[:, DIM_SURF] = self.insoluble_surface_spectrum.percentiles(
                np.random.random(n_sd)
            )
            simulated[:, DIM_TEMP] = self.freezing_temperature_spectrum.invcdf(
                np.random.random(n_sd), simulated[:, DIM_SURF]
            )

        return (
            simulated[:, DIM_TEMP],
            simulated[:, DIM_SURF],
            np.full((n_sd,), self.insoluble_surface_spectrum.norm_factor / n_sd),
        )
