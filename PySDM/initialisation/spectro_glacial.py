from PySDM.initialisation.spectral_sampling import SpectralSampling
from PySDM.physics import constants as const
import numpy as np

DIM_SIZE = 0
DIM_TEMP = 1
N_DIMS = 2

class SpectroGlacialSampling(SpectralSampling):
    pass

class Independent(SpectroGlacialSampling):
    def __init__(self,
                 size_spectrum, freezing_temperature_spectrum,
                 size_range=None, temperature_range=None,
                 seed=const.default_random_seed):
        super().__init__(size_spectrum, size_range)
        self.freezing_temperature_spectrum = freezing_temperature_spectrum
        self.temp_range = temperature_range
        self.rng = np.random.default_rng(seed)

    def sample(self, n_sd):
        pdf_arg = (
            self.rng.uniform(*self.size_range, n_sd),
            self.rng.uniform(*self.temp_range, n_sd)
        )
        dr = (self.size_range[1] - self.size_range[0]) / n_sd**(1/N_DIMS)
        dT = (self.temp_range[1] - self.temp_range[0]) / n_sd**(1/N_DIMS)
        return (
            *pdf_arg, dr * dT \
               * self.spectrum.size_distribution(pdf_arg[DIM_SIZE]) \
               * self.freezing_temperature_spectrum.pdf(pdf_arg[DIM_TEMP])
        )