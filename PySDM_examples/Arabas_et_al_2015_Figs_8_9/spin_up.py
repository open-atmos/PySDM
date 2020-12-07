"""
Created at 27.05.2020
"""

from PySDM.dynamics import Displacement
from PySDM.dynamics import Coalescence
import warnings


class SpinUp:

    def __init__(self, particles, spin_up_steps):
        self.spin_up_steps = spin_up_steps
        particles.observers.append(self)
        self.particles = particles
        self.set(Coalescence, 'enable', False)
        self.set(Displacement, 'enable_sedimentation', False)

    def notify(self):
        if self.particles.n_steps == self.spin_up_steps:
            self.set(Coalescence, 'enable', True)
            self.set(Displacement, 'enable_sedimentation', True)

    def set(self, dynamic, attr, value):
        key = dynamic.__name__
        if key in self.particles.dynamics:
            setattr(self.particles.dynamics[key], attr, value)
            # TODO: log it
        else:
            warnings.warn(f"{key} not found!")