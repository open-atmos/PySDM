from typing import Optional

import numpy as np
from pystrict import strict

from PySDM.dynamics.collisions.coalescence_efficiencies import ConstEc
from PySDM.dynamics.collisions.collision_kernels import Golovin
from PySDM.formulae import Formulae
from PySDM.initialisation import spectra
from PySDM.physics import si

TRIVIA = Formulae().trivia


@strict
class Settings0D:
    def __init__(
        self,
        seed: Optional[int] = None,
        warn_overflows: bool = True,
    ):
        self.n_sd = 2**12
        self.n_part = 100 / si.cm**3
        self.dv = 1 * si.m**3
        self.norm_factor = self.n_part * self.dv
        self.rho = 1000 * si.kilogram / si.metre**3
        self.dt = 1 * si.seconds
        self.adaptive = True
        self.warn_overflows = warn_overflows
        self.seed = 44
        self._steps = [0]
        self.kernel = Golovin(b=5e3 * si.s)
        self.coal_eff = ConstEc(Ec=1.0)
        self.vmin = 0.0
        self.spectrum = spectra.Gamma(
            norm_factor=self.norm_factor, k=1.0, theta=1e5 * si.um**3
        )
        self.radius_bins_edges = np.logspace(
            np.log10(1.0 * si.um), np.log10(5000 * si.um), num=64, endpoint=True
        )
        self.radius_range = [0 * si.um, 1e6 * si.um]
        self.formulae = Formulae(seed=seed)

    @property
    def output_steps(self):
        return [int(step / self.dt) for step in self._steps]
