from typing import Union
from PySDM_examples.Shima_et_al_2009.settings import Settings as Settings_Shima

from PySDM.dynamics.collisions.collision_kernels import Golovin, Geometric

from PySDM.physics import si
import numpy as np


class Settings(Settings_Shima):
    def __init__(self, n_sd=None, steps=None):
        super().__init__()

        self.n_sd = n_sd or 2**13
        self.steps = steps or [0, 1200, 2400, 3600]

        self.X0 = self.formulae.trivia.volume(radius=30.531 * si.micrometres)

        self.kernel: Union[Golovin, Geometric] = Golovin(b=1.5e3 / si.second)

        self.radius_bins_edges = np.logspace(
            np.log10(10 * si.um), np.log10(5e3 * si.um), num=128, endpoint=True
        )
