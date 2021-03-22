"""
Created at 08.08.2019
"""

import numpy as np
from PySDM.initialisation.spectra import Exponential
from PySDM.dynamics.coalescence.kernels import Golovin
from PySDM.physics.constants import si
from PySDM.physics.formulae import volume
from pystrict import strict


class Settings:

    def __init__(self):
        self.n_sd = 2 ** 13
        self.n_part = 2 ** 23 / si.metre**3
        self.X0 = volume(radius=30.531 * si.micrometres)
        self.dv = 1e6 * si.metres**3
        self.norm_factor = self.n_part * self.dv
        self.rho = 1000 * si.kilogram / si.metre**3
        self.dt = 1 * si.seconds
        self.adaptive = False
        self.seed = 44
        self._steps = [0, 1200, 2400, 3600]
        self.kernel = Golovin(b=1.5e3 / si.second)
        self.spectrum = Exponential(norm_factor=self.norm_factor, scale=self.X0)
        self.radius_bins_edges = np.logspace(np.log10(10 * si.um), np.log10(5e3 * si.um), num=128, endpoint=True)

    @property
    def output_steps(self):
        return [int(step/self.dt) for step in self._steps]
