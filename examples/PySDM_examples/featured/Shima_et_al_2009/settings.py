from typing import Optional

import numpy as np
from pystrict import strict

from PySDM import Formulae
from PySDM.dynamics.collisions.collision_kernels import Golovin
from PySDM.initialisation import spectra
from PySDM.physics import si


@strict
class Settings:
    def __init__(self, steps: Optional[list] = None):
        steps = steps or [0, 1200, 2400, 3600]
        self.formulae = Formulae()
        self.n_sd = 2**13
        self.n_part = 2**23 / si.metre**3
        self.X0 = self.formulae.trivia.volume(radius=30.531 * si.micrometres)
        self.dv = 1e6 * si.metres**3
        self.norm_factor = self.n_part * self.dv
        self.rho = 1000 * si.kilogram / si.metre**3
        self.dt = 1 * si.seconds
        self.adaptive = False
        self.seed = 44
        self.steps = steps
        self.kernel = Golovin(b=1.5e3 / si.second)
        self.spectrum = spectra.Exponential(norm_factor=self.norm_factor, scale=self.X0)
        self.radius_bins_edges = np.logspace(
            np.log10(10 * si.um), np.log10(5e3 * si.um), num=128, endpoint=True
        )

    @property
    def output_steps(self):
        return [int(step / self.dt) for step in self.steps]
