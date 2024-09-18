"""basic water vapor deposition on ice test"""

import numpy as np

import pytest

from PySDM.physics import si, diffusion_coordinate
from PySDM.backends import CPU
from PySDM import Builder
from PySDM import Formulae
from PySDM.environments import Box
from PySDM.dynamics import VapourDepositionOnIce
from PySDM.products import IceWaterContent


@pytest.mark.parametrize("dt", (1 * si.s, 0.1 * si.s))
@pytest.mark.parametrize("water_mass", (-si.ng, -si.ug, -si.mg, si.mg))
@pytest.mark.parametrize("fastmath", (True, False))
@pytest.mark.parametrize("diffusion_coordinate", ("Mass", "MassLogarithm"))
def test_iwc_lower_after_timestep(
    dt, water_mass, fastmath, diffusion_coordinate, dv=1 * si.m**3
):
    # arrange
    n_sd = 1
    builder = Builder(
        n_sd=n_sd,
        environment=Box(dt=dt, dv=dv),
        backend=CPU(
            formulae=Formulae(
                fastmath=fastmath,
                particle_shape_and_density="MixedPhaseSpheres",
                diffusion_coordinate=diffusion_coordinate,
            )
        ),
    )
    deposition = VapourDepositionOnIce()
    builder.add_dynamic(deposition)
    particulator = builder.build(
        attributes={
            "multiplicity": np.full(shape=(n_sd,), fill_value=1),
            "water mass": np.full(shape=(n_sd,), fill_value=water_mass),
        },
        products=(IceWaterContent(),),
    )
    particulator.environment["T"] = 250 * si.K
    particulator.environment["P"] = 500 * si.hPa
    particulator.environment["Schmidt number"] = 1

    # act
    iwc_old = particulator.products["ice water content"].get().copy()
    particulator.run(steps=1)
    iwc_new = particulator.products["ice water content"].get().copy()

    # assert
    assert (iwc_new > iwc_old).all() if water_mass < 0 else (iwc_new == iwc_old).all()
