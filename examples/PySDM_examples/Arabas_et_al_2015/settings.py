from typing import Iterable

import numpy as np
from PySDM_examples.Morrison_and_Grabowski_2007.strato_cumulus import StratoCumulus

from PySDM import Formulae
from PySDM.dynamics.collisions.breakup_efficiencies import ConstEb
from PySDM.dynamics.collisions.breakup_fragmentations import Gaussian
from PySDM.dynamics.collisions.coalescence_efficiencies import ConstEc
from PySDM.physics import si


class Settings(StratoCumulus):
    def __dir__(self) -> Iterable[str]:
        return (
            "dt",
            "grid",
            "size",
            "n_spin_up",
            "versions",
            "steps_per_output_interval",
            "formulae",
            "initial_dry_potential_temperature_profile",
            "initial_vapour_mixing_ratio_profile",
            "rhod_w_max",
        )

    def __init__(
        self,
        formulae=None,
        rhod_w_max: float = 0.6 * si.metres / si.seconds * (si.kilogram / si.metre**3),
    ):
        super().__init__(formulae or Formulae(), rhod_w_max=rhod_w_max)

        self.grid = (25, 25)
        self.size = (1500 * si.metres, 1500 * si.metres)

        # output steps
        self.simulation_time = 90 * si.minute
        self.dt = 5 * si.second
        self.spin_up_time = 1 * si.hour

        # additional breakup dynamics
        mu_r = 10 * si.um
        mu = 4 / 3 * np.pi * mu_r**3
        sigma = mu / 2.5
        vmin = mu / 1000
        self.coalescence_efficiency = ConstEc(Ec=0.95)
        self.breakup_efficiency = ConstEb(Eb=1.0)
        self.breakup_fragmentation = Gaussian(mu=mu, sigma=sigma, vmin=vmin, nfmax=10)
