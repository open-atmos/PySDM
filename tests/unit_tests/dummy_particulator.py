# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import numpy as np

from PySDM.particulator import Particulator

from .dummy_environment import DummyEnvironment


class DummyParticulator(Particulator):
    def __init__(
        self,
        backend_class=None,
        n_sd=0,
        formulae=None,
        grid=None,
        dynamics=None,
        attributes=None,
        requested_attributes=None,
        environment=None,
    ):
        if attributes is None:
            attributes = {
                "multiplicity": np.ones(n_sd),
                "signed water mass": np.full(n_sd, np.nan),
            }
        if requested_attributes is None:
            requested_attributes = ("cell id",)
        if environment is None:
            environment = DummyEnvironment(
                grid=grid,
                backend=backend_class(formulae, double_precision=True),
            )
        super().__init__(
            n_sd=n_sd,
            environment=environment,
            attributes=attributes,
            dynamics=dynamics,
            requested_attributes=requested_attributes,
        )
