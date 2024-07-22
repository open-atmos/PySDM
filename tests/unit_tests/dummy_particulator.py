# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from PySDM.builder import Builder
from PySDM.particulator import Particulator
from PySDM.attributes.numerics import CellId
from PySDM.attributes.physics import Multiplicity

from .dummy_environment import DummyEnvironment


class DummyParticulator(Builder, Particulator):
    def __init__(self, backend_class, n_sd=0, formulae=None, grid=None):
        backend = backend_class(formulae, double_precision=True)
        env = DummyEnvironment(grid=grid)
        Builder.__init__(self, n_sd, backend, env)
        Particulator.__init__(self, n_sd, backend)
        self.environment = env
        self.particulator = self
        self.req_attr = {"multiplicity": Multiplicity(self), "cell id": CellId(self)}
        self.attributes = None
