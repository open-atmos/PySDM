"""
test for collision kernel basics
"""

import numpy as np
import pytest
from PySDM import Builder
from PySDM.formulae import Formulae, _choices
from PySDM.physics import collision_kernel_liquid_liquid
from PySDM.environments import Box
from PySDM.dynamics import Coalescence
from PySDM.products import (
    LiquidWaterContent,
    ParticleConcentration,
)
from PySDM.physics import si


class TestCollisionKernel:
    @staticmethod
    @pytest.mark.parametrize("variant", _choices(collision_kernel_liquid_liquid))
    def test_collision_kernel_call(backend_class, variant):
        if (
            variant == "Linear"
            or variant == "ConstantK"
            and backend_class.__name__ == "ThrustRTC"
        ):
            pytest.skip()

        formulae = Formulae(
            collision_kernel_liquid_liquid=variant,
            seed=666,
            constants={
                "CONSTANTK_a": 1,
                "LINEAR_a": 1,
                "LINEAR_b": 1,
            },
        )
        env = Box(dt=1 * si.s, dv=1 * si.m**3)
        builder = Builder(
            n_sd=2,
            backend=backend_class(formulae=formulae),
            environment=env,
            dynamics=([Coalescence()]),
        )
        particulator = builder.build(
            products=(LiquidWaterContent(name="qc"), ParticleConcentration(name="nc")),
            attributes={
                "multiplicity": np.ones(builder.particulator.n_sd),
                "volume": np.ones(builder.particulator.n_sd) * si.mm**3,
            },
        )

        # act
        nc_initial = particulator.products["nc"].get()
        particulator.run(steps=1)

        # assert
        qc, nc = (particulator.products[k].get() for k in ("qc", "nc"))
        assert np.isfinite([qc, nc]).all() and qc > 0 and nc > 0
        assert nc <= nc_initial
