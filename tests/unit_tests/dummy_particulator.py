# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from PySDM import Builder, Particulator
from PySDM.attributes.numerics import CellID
from PySDM.attributes.physics import Multiplicities

from .dummy_environment import DummyEnvironment


class DummyParticulator(Builder, Particulator):
    def __init__(self, backend_class, n_sd=0, formulae=None, grid=None):
        backend = backend_class(formulae, double_precision=True)
        env = DummyEnvironment(grid=grid)
        Builder.__init__(self, n_sd, backend, env)
        Particulator.__init__(self, n_sd, backend)
        self.environment = env
        self.particulator = self
        self.req_attr = {"multiplicity": Multiplicities(self), "cell id": CellID(self)}
        self.attributes = None
