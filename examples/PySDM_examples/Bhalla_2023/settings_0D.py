from typing import Union, Optional
from PySDM_examples.Shima_et_al_2009.settings import Settings as Settings_Shima

from PySDM.dynamics.collisions.collision_kernels import Golovin, Geometric

from PySDM.physics import si
import numpy as np


class Settings(Settings_Shima):
    def __init__(self, n_sd: Union[int, None] = None, max_t: Optional[int] = None, n_part: Optional[float] = None, evaluate_relaxed_velocity=False, tau=1*si.second):
        super().__init__()

        self.n_sd = n_sd or 2**13
        self.max_t = max_t or 3600
        self.steps = np.arange(0, self.max_t+1, 50)

        self.X0 = self.formulae.trivia.volume(radius=30.531 * si.micrometres)

        # self.kernel: Union[Golovin, Geometric] = Golovin(b=1.5e3 / si.second)
        self.kernel: Union[Golovin, Geometric] = Geometric(relax_velocity=evaluate_relaxed_velocity)

        self.radius_bins_edges = np.logspace(
            np.log10(10 * si.um), np.log10(5e4 * si.um), num=128, endpoint=True
        )

        self.n_part = n_part or 2**23 / si.metre**3

        self.evaluate_relaxed_velocity = evaluate_relaxed_velocity
        self.tau = tau
