"""
test for collision kernel basics
"""

import numpy as np
import pytest
from PySDM import Builder, Formulae
from PySDM.environments import Box
from PySDM.dynamics import Coalescence
from PySDM.dynamics.collisions.collision_kernels import Golovin
from PySDM.products import (
    LiquidWaterContent,
)
from PySDM.physics import si


class TestCollisionKernel:
    @staticmethod
    def test_collision_kernel_call(backend_class):
        if backend_class.__name__ == "ThrustRTC":
            pytest.skip()

        formulae = Formulae(
            collision_kernel_liquid_liquid="Golovin",
        )
        env = Box(dt=1 * si.s, dv=1 * si.m**3)
        builder = Builder(
            n_sd=2,
            backend=backend_class(formulae=formulae),
            environment=env,
            dynamics=(
                [
                    Coalescence(
                        collision_kernel=Golovin(b=5e3 * si.s),
                    )
                ]
            ),
        )
        particulator = builder.build(
            products=(LiquidWaterContent(name="qc"),),
            attributes={
                "multiplicity": np.ones(builder.particulator.n_sd),
                "volume": np.ones(builder.particulator.n_sd) * si.um**3,
            },
        )

        # act
        particulator.run(steps=1)

        # assert
        (qc,) = particulator.products["qc"].get()
        assert np.isfinite(qc)
