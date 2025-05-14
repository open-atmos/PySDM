from typing import Dict, Iterable, Optional
import numpy as np
from .strato_cumulus_seeding import StratoCumulus

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
            "particles_per_volume_STP",
            "n_sd_per_gridbox",
            "radius",
            "kappa",
            "z_part",
            "x_part",
            "simulation_time",
            "spin_up_time",
        )

    def __init__(
        self,
        formulae=None,
        rhod_w_max: float = 0.6 * si.metres / si.seconds * (si.kilogram / si.metre**3),
        grid: tuple = None,
        size: tuple = None,
        particles_per_volume_STP: int = 50 / si.cm**3,
        n_sd_per_gridbox: int = 32,
        radius: float = 0.04 * si.micrometre,
        kappa: float = 0.3,
        z_part: Optional[tuple] = None,
        x_part: Optional[tuple] = None,
        simulation_time: float = None,
        dt: float = None,
        spin_up_time: float = None,
    ):
        super().__init__(
            formulae or Formulae(),
            rhod_w_max=rhod_w_max,
            particles_per_volume_STP=particles_per_volume_STP,
            n_sd_per_gridbox=n_sd_per_gridbox,
            radius=radius,
            kappa=kappa,
        )

        self.grid = grid
        self.size = size
        self.particles_per_volume_STP = particles_per_volume_STP
        self.n_sd_per_gridbox = n_sd_per_gridbox
        self.radius = radius
        self.kappa = kappa
        self.z_part = z_part
        self.x_part = x_part

        # output steps
        self.simulation_time = simulation_time  # 90 * si.minute
        self.dt = dt  # 5 * si.second
        self.spin_up_time = spin_up_time  # 1 * si.hour

        # additional breakup dynamics
        mu_r = 10 * si.um
        mu = 4 / 3 * np.pi * mu_r**3
        sigma = mu / 2.5
        vmin = mu / 1000
        self.coalescence_efficiency = ConstEc(Ec=0.95)
        self.breakup_efficiency = ConstEb(Eb=1.0)
        self.breakup_fragmentation = Gaussian(mu=mu, sigma=sigma, vmin=vmin, nfmax=10)
