"""
Created at 25.11.2019

@author: Michael Olesik
@author: Sylwester Arabas
"""

from PySDM.simulation.initialisation.spectra import Lognormal
from PySDM.backends.default import Default
from PySDM.simulation.physics.constants import si
import numpy as np


class Setup:
    def __init__(self, dt=1 * si.second):
        self.dt = dt
        self.n_steps = int(3 * si.hours / dt)

    backend = Default

    mass_of_dry_air = 100 * si.kilogram
    spectrum = Lognormal(
      norm_factor=1000 / si.milligram * mass_of_dry_air,
      m_mode=50 * si.nanometre,
      s_geom=1.4
    )

    n_sd = 100
    T0 = 284.3 * si.kelvin
    q0 = 7.6 * si.grams / si.kilogram
    p0 = 938.5 * si.hectopascals
    z0 = 600 * si.metres
    kappa = 0.53  # Petters and S. M. Kreidenweis mean growth-factor derived

    # initial dry radius discretisation range
    r_min = 10.633 * si.nanometre
    r_max = 513.06 * si.nanometre

    @staticmethod
    def w(t):
        t0 = 1200 * si.second
        f0 = 1 / 1000 * si.hertz
        return .5 * (np.where(t < t0, 1, np.sign(-np.sin(2*np.pi * f0 * (t-t0))))) * si.metre / si.second
