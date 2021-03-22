"""
Created at 25.11.2019
"""

from PySDM.initialisation.spectra import Lognormal
from PySDM.backends import CPU
from PySDM.physics.constants import si
from PySDM.initialisation import spectral_sampling
from PySDM.dynamics import condensation
import numpy as np
from pystrict import strict


@strict
class Settings:

    def __init__(self, n_sd: int=100, dt_output: float=1 * si.second, dt_max: float=1 * si.second):
        self.total_time = 3 * si.hours
        self.mass_of_dry_air = 1000 * si.kilogram  # TODO #335 doubled with jupyter si unit

        self.n_steps = int(self.total_time / (5 * si.second))  # TODO #334 rename to n_output
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

        self.backend = CPU
        self.coord = 'volume logarithm'
        self.adaptive = True
        self.rtol_x = condensation.default_rtol_x
        self.rtol_thd = condensation.default_rtol_thd
        self.enable_particle_temperatures = False

        self.T0 = 284.3 * si.kelvin
        self.q0 = 7.6 * si.grams / si.kilogram
        self.p0 = 938.5 * si.hectopascals
        self.z0 = 600 * si.metres
        self.kappa = 0.53  # Petters and S. M. Kreidenweis mean growth-factor derived

        self.t0 = 1200 * si.second
        self.f0 = 1 / 1000 * si.hertz

    def w(self, t):
        return .5 * (np.where(t < self.t0, 1, np.sign(-np.sin(2 * np.pi * self.f0 * (t - self.t0))))) \
               * si.metre / si.second
