# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import numpy as np

from PySDM.particulator import Particulator

from .dummy_environment import DummyEnvironment


class DummyParticulator(Particulator):
    def __init__(self, backend_class, n_sd=0, formulae=None, grid=None, dynamics=None):
        Particulator.__init__(
            self,
            n_sd=n_sd,
            environment=DummyEnvironment(
                grid=grid, backend=backend_class(formulae, double_precision=True)
            ),
            attributes={
                "multiplicity": np.ones(n_sd),
                "signed water mass": np.full(n_sd, np.nan),
            },
            dynamics=dynamics,
            requested_attributes=("cell id",),
        )
