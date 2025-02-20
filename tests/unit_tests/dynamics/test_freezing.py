"""basic freezing of liquid drolpets test"""

import numpy as np

import pytest

from PySDM.physics import si
from PySDM.backends import CPU
from PySDM import Builder
from PySDM import Formulae
from PySDM.environments import Box
from PySDM.dynamics import Freezing
from PySDM.products import IceWaterContent





@pytest.mark.parametrize("dt", (1 * si.s,))
@pytest.mark.parametrize("water_mass", (si.mg,))
@pytest.mark.parametrize("RHi", (1.1,))



def test_iwc_lower_after_timestep(
    dt, water_mass, RHi, dv=1 * si.m**3
):
    # arrange
    n_sd = 1
    builder = Builder(
        n_sd=n_sd,
        environment=Box(dt=dt, dv=dv),
        backend=CPU(
            formulae=Formulae(
                fastmath=False,
                particle_shape_and_density="MixedPhaseSpheres",
            )
        ),
    )


    water_freezing = Freezing
    builder.add_dynamic(water_freezing)
    particulator = builder.build(
        attributes={
            "multiplicity": np.full(shape=(n_sd,), fill_value=1),
            "water mass": np.full(shape=(n_sd,), fill_value=water_mass),
        },
        products=(IceWaterContent(),LiquidWaterContent(),),
    )
    temperature = 250 * si.K
    particulator.environment["T"] = temperature
    pressure = 500 * si.hPa
    particulator.environment["P"] = pressure


    iwc = particulator.products["ice water content"].get().copy()
    lwc = particulator.products["liquid water content"].get().copy()


    assert (iwc > 0.).all() 
