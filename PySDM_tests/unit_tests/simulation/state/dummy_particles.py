"""
Created at 19.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.particles_builder import ParticlesBuilder
from PySDM.simulation.particles import Particles
from PySDM.simulation.mesh import Mesh


class DummyParticles(ParticlesBuilder, Particles):

    def __init__(self, backend, n_sd, dt=None):
        Particles.__init__(self, n_sd, dt, backend)
        # super(ParticlesBuilder, self).__init__(n_sd, dt, backend)
        self.particles = self
        self.environment = None

    def set_environment(self, environment_class, params):
        self.environment = environment_class(None, **params)
