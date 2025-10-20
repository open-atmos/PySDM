from typing import Iterable

from PySDM_examples.Morrison_and_Grabowski_2007.strato_cumulus import StratoCumulus

from PySDM import Formulae
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
        # self.simulation_time = 150 * si.minute
        # self.dt = 5 * si.second
        self.spin_up_time = 1 * si.hour

from PySDM.dynamics import Collision, Displacement


class SpinUp:
    def __init__(self, particulator, spin_up_steps):
        self.spin_up_steps = spin_up_steps
        particulator.observers.append(self)
        self.particulator = particulator
        self.set(Collision, "enable", False)
        self.set(Displacement, "enable_sedimentation", False)

    def notify(self):
        if self.particulator.n_steps == self.spin_up_steps:
            self.set(Collision, "enable", True)
            self.set(Displacement, "enable_sedimentation", True)
            # resample the particles
            

    def set(self, dynamic, attr, value):
        key = dynamic.__name__
        if key in self.particulator.dynamics:
            setattr(self.particulator.dynamics[key], attr, value)

