"""checks if @register_product makes dynamics instances reusable"""

import numpy as np

from PySDM import Particulator
from PySDM.backends import CPU
from PySDM.environments import Box
from PySDM.dynamics.impl import register_dynamic


def test_impl_register_dynamic():
    # arrange
    @register_dynamic()
    class Dynamic:  # pylint: disable=too-few-public-methods
        def __init__(self):
            self.particulator = None

        def register(self, *, particulator: Particulator):
            self.particulator = particulator

    dynamic = Dynamic()
    n_sd = 1
    kwargs = {
        "n_sd": n_sd,
        "environment": Box(dt=0, dv=0, backend=CPU()),
        "dynamics": (dynamic,),
        "attributes": {"multiplicity": np.ones(n_sd), "water mass": np.zeros(n_sd)},
    }

    # act
    particulators = (Particulator(**kwargs), Particulator(**kwargs))

    # assert
    assert dynamic.particulator is None
    for particulator in particulators:
        assert particulator.dynamics["Dynamic"].particulator is particulator
