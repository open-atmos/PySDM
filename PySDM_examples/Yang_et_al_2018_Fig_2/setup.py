"""
Created at 25.11.2019

@author: Michael Olesik
@author: Sylwester Arabas
"""

from PySDM.simulation.initialisation.spectra import Lognormal
from PySDM.backends.default import Default
from PySDM.simulation.physics.constants import si
from PySDM.simulation.initialisation import spectral_sampling
from PySDM.simulation.initialisation.multiplicities import discretise_n
from PySDM.simulation.dynamics.condensation import condensation
import numpy as np


class Setup:
    def __init__(self, n_sd=100, dt_output = 1 * si.second, dt_max=1 * si.second):
        self.n_steps = int(self.total_time / (5 * si.second) )  # TODO: rename to n_output
        self.n_sd = n_sd
        self.r_dry, self.n = spectral_sampling.logarithmic(
            n_sd=n_sd,
            spectrum=Lognormal(
                norm_factor=1000 / si.milligram * self.mass_of_dry_air,
                m_mode=50 * si.nanometre,
                s_geom=1.4
            ),
            range=(10.633 * si.nanometre, 513.06 * si.nanometre)
        )
        self.n = discretise_n(self.n)
        self.dt_output = dt_output
        self.dt_max = dt_max
    backend = Default
    coord = 'volume logarithm'
    adaptive = True
    rtol_x = condensation.default_rtol_x
    rtol_thd = condensation.default_rtol_thd

    mass_of_dry_air = 100 * si.kilogram
    total_time = 3 * si.hours
    T0 = 284.3 * si.kelvin
    q0 = 7.6 * si.grams / si.kilogram
    p0 = 938.5 * si.hectopascals
    z0 = 600 * si.metres
    kappa = 0.53  # Petters and S. M. Kreidenweis mean growth-factor derived

    @staticmethod
    def w(t):
        t0 = 1200 * si.second
        f0 = 1 / 1000 * si.hertz
        return .5 * (np.where(t < t0, 1, np.sign(-np.sin(2*np.pi * f0 * (t-t0))))) * si.metre / si.second
