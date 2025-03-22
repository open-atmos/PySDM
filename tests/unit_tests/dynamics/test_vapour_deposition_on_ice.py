"""basic water vapor deposition on ice test"""

import numpy as np

import pytest

from PySDM.physics import si
from PySDM.backends import CPU
from PySDM import Builder
from PySDM import Formulae
from PySDM.environments import Box
from PySDM.dynamics import VapourDepositionOnIce
from PySDM.products import IceWaterContent


@pytest.mark.parametrize("dt", (1 * si.s, 0.1 * si.s))
@pytest.mark.parametrize("water_mass", (-si.ng, -si.ug, -si.mg, si.mg))
@pytest.mark.parametrize("RHi", (1.1, 1.0, 0.9))
@pytest.mark.parametrize("fastmath", (True, False))
@pytest.mark.parametrize("diffusion_coordinate", ("WaterMass", "WaterMassLogarithm"))
def test_iwc_lower_after_timestep(
    *, dt, water_mass, RHi, fastmath, diffusion_coordinate, dv=1 * si.m**3
):
    # arrange
    builder = Builder(
        n_sd=1,
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
            "multiplicity": np.full(shape=(builder.particulator.n_sd,), fill_value=1),
            "signed water mass": np.full(
                shape=(builder.particulator.n_sd,), fill_value=water_mass
            ),
        },
        products=(IceWaterContent(),),
    )
    particulator.environment["T"] = 250 * si.K
    particulator.environment["p"] = 500 * si.hPa
    pvs_ice = particulator.formulae.saturation_vapour_pressure.pvs_ice(
        particulator.environment["T"][0]
    )
    pvs_water = particulator.formulae.saturation_vapour_pressure.pvs_water(
        particulator.environment["T"][0]
    )
    vapour_pressure = RHi * pvs_ice
    particulator.environment["RH"] = vapour_pressure / pvs_water
    particulator.environment["a_w_ice"] = pvs_ice / pvs_water
    particulator.environment["Schmidt number"] = 1
    rv0 = (
        particulator.formulae.constants.eps
        * vapour_pressure
        / (particulator.environment["p"][0] - vapour_pressure)
    )
    particulator.environment["water_vapour_mixing_ratio"] = rv0
    particulator.environment["rhod"] = (
        particulator.environment["p"][0] - vapour_pressure
    ) / (particulator.environment["T"][0] * particulator.formulae.constants.Rd)

    # act
    temperature = particulator.environment["T"][0]
    iwc_old = particulator.products["ice water content"].get().copy()
    particulator.run(steps=1)

    # assert
    if water_mass < 0 and RHi != 1:
        if RHi > 1:
            assert (particulator.products["ice water content"].get() > iwc_old).all()
            assert particulator.environment["water_vapour_mixing_ratio"][0] < rv0
            assert particulator.environment["T"][0] > temperature
        elif RHi < 1:
            assert (particulator.products["ice water content"].get() < iwc_old).all()
            assert particulator.environment["water_vapour_mixing_ratio"][0] > rv0
            assert particulator.environment["T"][0] < temperature
    else:
        assert (particulator.products["ice water content"].get() == iwc_old).all()
        assert particulator.environment["water_vapour_mixing_ratio"][0] == rv0
        assert particulator.environment["T"][0] == temperature
