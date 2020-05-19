"""
Created at 19.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.particles_builder import ParticlesBuilder
from PySDM.particles import Particles
from .dummy_environment import DummyEnvironment
from PySDM.attributes.droplet.multiplicities import Multiplicities
from PySDM.attributes.droplet.volume import Volume
from PySDM.attributes.cell.cell_id import CellID


class DummyParticles(ParticlesBuilder, Particles):

    def __init__(self, backend, n_sd):
        Particles.__init__(self, n_sd, backend)
        self.particles = self
        self.environment = DummyEnvironment(self)
        self.req_attr = {'n': Multiplicities(self), 'cell id': CellID(self)}
        self.state = None


    def set_environment(self, environment_class, params):
        self.environment = environment_class(None, **params)
