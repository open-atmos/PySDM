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
    backend = Default

    mass = 1 * si.kilogram
    spectrum = Lognormal(
      norm_factor=1000 / si.milligram / si.nanometre * mass,
      m_mode=50 * si.nanometre,
      s_geom=1.4
    )

    n_sd = 100
    T0 = 284.3 * si.kelvin
    q0 = 7.6 * si.grams / si.kilogram
    p0 = 938.5 * si.hectopascals
    kappa = 0.53 # Petters and S. M. Kreidenweis mean growth-factor derived

    # initial dry radius discretisation range
    r_min = 10.633 * si.nanometre
    r_max = 513.06 * si.nanometre

    dt = 0.1 * si.second

    @staticmethod
    def w(t):
        t0 = 1200 * si.second
        f0 = 1 / 1000 * si.hertz
        return .5 * (np.where(t < t0, 1, np.sign(-np.sin(2*np.pi * f0 * (t-t0))))) * si.metre / si.second
