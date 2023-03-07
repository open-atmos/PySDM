"""
Marshall-Palmer spectrum
"""
import numpy as np
from scipy.interpolate import interp1d

from PySDM.formulae import Formulae
from PySDM.initialisation.sampling.spectral_sampling import default_cdf_range
from PySDM.physics.constants import si

default_interpolation_grid = tuple(np.linspace(*default_cdf_range, 999))
MP_N0 = 8000 * si.m ** (-3) * si.mm ** (-1)
MP_LAMBDA0 = 4.1 * si.mm ** (-1)
MP_LAMBDA1 = -0.21
TRIVIA = Formulae().trivia


class MarshallPalmer:
    def __init__(self, rain_rate, dv, interpolation_grid=None):
        self.rain_rate = rain_rate / si.mm * si.h
        self.scale = MP_LAMBDA0 * self.rain_rate ** (MP_LAMBDA1) * si.mm ** (-1)
        self.norm_factor = MP_LAMBDA0 * dv / self.scale
        # interpolation_grid = interpolation_grid or default_interpolation_grid
        # cdf_arg = np.zeros(len(interpolation_grid) + 1)
        # cdf_arg[1:] = interpolation_grid
        # cdf = self.cumulative(cdf_arg) / self.norm_factor
        # print(cdf, cdf_arg)
        # self.inverse_cdf = interp1d(cdf, cdf_arg)

    def size_distribution(self, arg):
        diam_arg = 2 * TRIVIA.radius(volume=arg)
        result = MP_N0 * np.exp(-self.scale * diam_arg)
        return result

    def cumulative(self, arg):
        diam_arg = 2 * TRIVIA.radius(volume=arg)
        cdf = 1.0 - np.exp(-self.scale * diam_arg)
        return self.norm_factor * cdf

    def percentiles(self, cdf_values):
        print(cdf_values[0], cdf_values[-1])
        diams = np.array(
            [-np.log(1 - cdf_values[i]) / self.scale for i in range(len(cdf_values))]
        )
        print(diams[0], diams[-1])
        return TRIVIA.volume(radius=diams / 2)
