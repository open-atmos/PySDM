"""
Created at 08.08.2019
"""

import numpy as np
from PySDM.initialisation.spectra import Exponential
from PySDM.dynamics.coalescence.kernels import Geometric
from PySDM.physics.constants import si
from PySDM.physics import formulae as phys
from pystrict import strict


@strict
class Settings:
    def __init__(self):
        self.init_x_min = phys.volume(radius=3.94 * si.micrometre)
        self.init_x_max = phys.volume(radius=25 * si.micrometres)

        self.n_sd = 2 ** 13
        self.n_part = 239 / si.cm**3
        self.X0 = phys.volume(radius=10 * si.micrometres)
        self.dv = 1e1 * si.metres**3  # TODO #336 1e6 do not work with ThrustRTC (overflow?)
        self.norm_factor = self.n_part * self.dv
        self.rho = 1000 * si.kilogram / si.metre**3
        self.dt = 1 * si.seconds
        self.adaptive = False
        self.seed = 44
        self._steps = [200 * i for i in range(10)]
        self.kernel = Geometric(collection_efficiency=1)
        self.spectrum = Exponential(norm_factor=self.norm_factor, scale=self.X0)

        # Note 220 instead of 200 to smoothing
        self.radius_bins_edges = np.logspace(np.log10(3.94 * si.um), np.log10(220 * si.um), num=100, endpoint=True)

    @property
    def output_steps(self):
        return [int(step / self.dt) for step in self._steps]

