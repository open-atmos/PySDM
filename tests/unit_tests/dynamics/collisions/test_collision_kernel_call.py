"""
test for collision kernel basics
"""

import numpy as np
import pytest
from PySDM import Particulator
from PySDM.formulae import Formulae, _choices
from PySDM.physics import collision_kernel
from PySDM.environments import Box
from PySDM.dynamics import Coalescence
from PySDM.products import (
    LiquidWaterContent,
    ParticleConcentration,
)
from PySDM.physics import si


class TestCollisionKernel:  # pylint: disable=too-few-public-methods
    @staticmethod
    @pytest.mark.parametrize("variant", _choices(collision_kernel))
    def test_collision_kernel_call(backend_class, variant):
        if variant == "Null":
            pytest.skip()

        formulae = Formulae(
            collision_kernel=variant,
            seed=666,
            constants={
                "KERNEL_GOLOVIN_b": 0,
                "KERNEL_CONSTANTK_a": 1,
                "KERNEL_LINEAR_a": 1,
                "KERNEL_LINEAR_b": 1,
                "KERNEL_SIMPLE_GEOMETRIC_c": 1,
            },
        )
        env = Box(dt=1 * si.s, dv=1 * si.m**3, backend=backend_class(formulae=formulae))
        particulator = Particulator(
            n_sd=(n_sd := 2),
            environment=env,
            dynamics=([Coalescence()]),
            products=(LiquidWaterContent(name="qc"), ParticleConcentration(name="nc")),
            attributes={
                "multiplicity": np.ones(n_sd),
                "volume": np.ones(n_sd) * si.mm**3,
            },
        )

        # act
        nc_initial = particulator.products["nc"].get()
        particulator.run(steps=1)

        # assert
        qc, nc = (particulator.products[k].get() for k in ("qc", "nc"))
        assert np.isfinite([qc, nc]).all() and qc > 0 and nc > 0
        assert nc <= nc_initial
