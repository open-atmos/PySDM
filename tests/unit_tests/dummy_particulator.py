# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from PySDM import Builder, Particulator
from PySDM.physics import Formulae
from PySDM.attributes.physics import Multiplicities
from PySDM.attributes.numerics import CellID
from .dummy_environment import DummyEnvironment


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
