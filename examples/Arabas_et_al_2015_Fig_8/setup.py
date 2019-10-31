"""
Created at 25.09.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
import pint
from SDM.simulation.spectra import Lognormal
from SDM.simulation.kernels import Golovin
from SDM.backends.default import Default


class Setup:
    si = pint.UnitRegistry()

    grid = (75, 75)  # dx=dz=20m
    size = (1500, 1500)  # [m]

    @property
    def dv(self):
        return self.size[0] / self.grid[0] * self.size[1] / self.grid[1]  # [m3]

    @property
    def n_sd(self):
        return self.grid[0] * self.grid[1] * 2

    # TODO: second mode
    # TODO: number -> mass distribution
    spectrum = Lognormal(
      norm_factor=(40 / si.centimetre**3 * size[0]*si.metre * size[1]*si.metre * 1*si.metre).to_base_units().magnitude, 
      m_mode=(0.15 * si.micrometre).to_base_units().magnitude,
      s_geom=1.6
    )

    processes = {
        "advection": True,
        "coalescence": False
    }

    field_values = {'th': 300,
                    'qv': 10e-3}

    def stream_function(self, x, z):
        w_max = .6
        X = self.size[0]
        Z = self.size[1]
        return - w_max * X / np.pi * np.sin(np.pi * z / Z) * np.cos (2 * np.pi * x / X)

    x_min = .01e-6 # TODO: mass!
    x_max = 5e-6 # TODO: mass!

    dt = 1  # [s]

    steps = np.arange(0, 3600, 30)

    kernel = Golovin(b=1.5e3)  # [s-1]

    backend = Default()
