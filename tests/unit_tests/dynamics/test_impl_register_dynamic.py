""" checks if @register_product makes dynamics instances reusable """

import numpy as np

from PySDM import Builder
from PySDM.backends import CPU
from PySDM.environments import Box
from PySDM.dynamics.impl import register_dynamic


def test_impl_register_dynamic():
    # arrange
    @register_dynamic()
    class Dynamic:
        def __init__(self):
            self.particulator = None

        def register(self, *, builder: Builder):
            self.particulator = builder.particulator

    dynamic = Dynamic()
    kwargs = {"n_sd": 0, "backend": CPU(), "environment": Box(dt=0, dv=0)}
    builders = [Builder(**kwargs), Builder(**kwargs)]

    # act
    for builder in builders:
        builder.add_dynamic(dynamic)
        builder.build(
            attributes={"multiplicity": np.empty(0), "water mass": np.empty(0)}
        )

    # assert
    assert dynamic.particulator is None
    assert builders[0].particulator is not builders[1].particulator
    for builder in builders:
        assert (
            builder.particulator.dynamics["Dynamic"].particulator
            is builder.particulator
        )
