"""
Created at 25.11.2019
"""

from PySDM.initialisation.spectra import Lognormal
from PySDM.backends import CPU
from PySDM.physics.constants import si
from PySDM.initialisation import spectral_sampling
from PySDM.dynamics import condensation
import numpy as np


class Settings:

    def __init__(self, n_sd=100, dt_output = 1 * si.second, dt_max=1 * si.second):
        self.n_steps = int(self.total_time / (5 * si.second) )  # TODO: rename to n_output
        self.n_sd = n_sd
        self.r_dry, self.n = spectral_sampling.Logarithmic(
            spectrum=Lognormal(
                norm_factor=1000 / si.milligram * self.mass_of_dry_air,
                m_mode=50 * si.nanometre,
                s_geom=1.4
            ),
            size_range=(10.633 * si.nanometre, 513.06 * si.nanometre)
        ).sample(n_sd)
        self.dt_max = dt_max

        self.dt_output = dt_output
        self.r_bins_edges = np.linspace(0 * si.micrometre, 20 * si.micrometre, 101, endpoint=True)

    backend = CPU
    coord = 'volume logarithm'
    adaptive = True
    rtol_x = condensation.default_rtol_x
    rtol_thd = condensation.default_rtol_thd

    mass_of_dry_air = 100 * si.kilogram  # TODO: doubled with jupyter si unit
    total_time = 3 * si.hours
    T0 = 284.3 * si.kelvin
    q0 = 7.6 * si.grams / si.kilogram
    p0 = 938.5 * si.hectopascals
    z0 = 600 * si.metres
    kappa = 0.53  # Petters and S. M. Kreidenweis mean growth-factor derived

    t0 = 1200 * si.second
    f0 = 1 / 1000 * si.hertz

    @staticmethod
    def w(t):
        return .5 * (np.where(t < Settings.t0, 1, np.sign(-np.sin(2 * np.pi * Settings.f0 * (t - Settings.t0))))) \
               * si.metre / si.second
