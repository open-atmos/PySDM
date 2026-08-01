# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from PySDM.particulator import Particulator

from .dummy_environment import DummyEnvironment


class DummyParticulator(Particulator):
    def __init__(self, backend_class, n_sd=0, formulae=None, grid=None, dynamics=None):
        backend = backend_class(formulae, double_precision=True)
        env = DummyEnvironment(grid=grid)
        Particulator.__init__(
            self,
            n_sd,
            backend,
            environment=env,
            attributes={},
            dynamics=dynamics,
        )
        self.environment = env.instantiate(builder=self)  # pylint: disable=no-member
        self.dynamics = self.particulator.dynamics
        self.particulator = self
        self.req_attr_names = ["multiplicity", "cell id"]
        self.attributes = None
