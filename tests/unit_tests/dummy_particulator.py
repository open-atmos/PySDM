# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from PySDM import Builder, Particulator
from PySDM.attributes.physics import Multiplicities
from PySDM.attributes.numerics import CellID
from .dummy_environment import DummyEnvironment


class DummyParticulator(Builder, Particulator):

    def __init__(self, backend_class, n_sd=0, formulae=None):
        backend = backend_class(formulae)
        Builder.__init__(self, n_sd, backend)
        Particulator.__init__(self, n_sd, backend)
        self.particulator = self
        self.environment = DummyEnvironment()
        self.environment.register(self)
        self.req_attr = {'n': Multiplicities(self), 'cell id': CellID(self)}
        self.attributes = None
