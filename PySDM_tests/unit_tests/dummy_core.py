"""
Created at 19.11.2019
"""

from PySDM.builder import Builder
from PySDM.core import Core
from .dummy_environment import DummyEnvironment
from PySDM.attributes.droplet.multiplicities import Multiplicities
from PySDM.attributes.cell.cell_id import CellID


class DummyCore(Builder, Core):

    def __init__(self, backend, n_sd=0):
        Core.__init__(self, n_sd, backend)
        self.core = self
        self.environment = DummyEnvironment()
        self.environment.register(self)
        self.req_attr = {'n': Multiplicities(self), 'cell id': CellID(self)}
        self.particles = None
