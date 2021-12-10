from PySDM.builder import Builder
from PySDM.physics.formulae import Formulae
from PySDM.particulator import Particulator
from .dummy_environment import DummyEnvironment
from PySDM.attributes.physics.multiplicities import Multiplicities
from PySDM.attributes.numerics.cell_id import CellID


class DummyParticulator(Builder, Particulator):

    def __init__(self, backend, n_sd=0, formulae=None):
        if formulae is None:
            formulae = Formulae()
        Particulator.__init__(self, n_sd, backend(formulae))
        self.particulator = self
        self.environment = DummyEnvironment()
        self.environment.register(self)
        self.req_attr = {'n': Multiplicities(self), 'cell id': CellID(self)}
        self.attributes = None
