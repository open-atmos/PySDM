"""
Created at 27.05.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.dynamics import Displacement
from PySDM.dynamics import Coalescence


class SpinUp:
    def __init__(self, particles, spin_up_steps):
        self.spin_up_steps = spin_up_steps
        particles.observers.append(self)
        self.particles = particles
        self.set(str(Coalescence), 'enable', False)
        self.set(str(Displacement), 'enable_sedimentation', False)

    def notify(self):
        if self.particles.n_steps == self.spin_up_steps:
            self.set(str(Coalescence), 'enable', True)
            self.set(str(Displacement), 'enable_sedimentation', True)

    def set(self, key, attr, value):
        if key in self.particles.dynamics:
            setattr(self.particles.dynamics[key], attr, value)
