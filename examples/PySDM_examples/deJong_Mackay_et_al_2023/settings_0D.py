from typing import Optional

import numpy as np
from pystrict import strict

from PySDM.dynamics.collisions import breakup_fragmentations
from PySDM.dynamics.collisions.breakup_efficiencies import ConstEb
from PySDM.dynamics.collisions.coalescence_efficiencies import Berry1967
from PySDM.dynamics.collisions.collision_kernels import Geometric
from PySDM.formulae import Formulae
from PySDM.initialisation import spectra
from PySDM.physics import si

TRIVIA = Formulae().trivia


@strict
class Settings0D:
    X0 = TRIVIA.volume(radius=30.531 * si.micrometres)

    def __init__(
        self,
        fragmentation: object = None,
        seed: Optional[int] = None,
        warn_overflows: bool = True,
    ):
        self.n_sd = 2**10
        self.n_part = 100 / si.cm**3
        self.frag_scale = TRIVIA.volume(radius=100 * si.micrometres)
        self.dv = 1 * si.m**3
        self.norm_factor = self.n_part * self.dv
        self.rho = 1000 * si.kilogram / si.metre**3
        self.dt = 1 * si.seconds
        self.adaptive = True
        self.warn_overflows = warn_overflows
        self.seed = 44
        self._steps = [0]
        self.kernel = Geometric()
        self.coal_eff = Berry1967()
        self.fragmentation = fragmentation or breakup_fragmentations.Exponential(
            scale=self.frag_scale
        )
        self.vmin = 0.0
        self.break_eff = ConstEb(1.0)  # no "bouncing"
        self.spectrum = spectra.Exponential(norm_factor=self.norm_factor, scale=self.X0)
        self.radius_bins_edges = np.logspace(
            np.log10(0.01 * si.um), np.log10(5000 * si.um), num=64, endpoint=True
        )
        self.radius_range = [0 * si.um, 1e6 * si.um]
        self.formulae = Formulae(
            seed=seed, fragmentation_function=self.fragmentation.__class__.__name__
        )

    @property
    def output_steps(self):
        return [int(step / self.dt) for step in self._steps]
