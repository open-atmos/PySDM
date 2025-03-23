"""basic water vapor deposition on ice test"""

import numpy as np

import pytest

from PySDM.physics import si
from PySDM.backends import CPU
from PySDM import Builder
from PySDM import Formulae
from PySDM.environments import Box
from PySDM.environments.impl import register_environment
from PySDM.environments.impl.moist import Moist
from PySDM.dynamics import VapourDepositionOnIce, AmbientThermodynamics
from PySDM.products import IceWaterContent


@register_environment()
class MoistBox(Box, Moist):
    def __init__(self, dt: float, dv: float, mixed_phase: bool = False):
        Box.__init__(self, dt, dv)
        Moist.__init__(self, dt, self.mesh, variables=["rhod"], mixed_phase=mixed_phase)

    def register(self, builder):
        Moist.register(self, builder)

    def get_water_vapour_mixing_ratio(self):
        return self["water_vapour_mixing_ratio"]

    def get_thd(self):
        return self["thd"]

    def sync(self):
        Moist.sync(self)

    def get_predicted(self, key: str):
        return self[key]


@pytest.mark.parametrize("dt", (1 * si.s, 0.1 * si.s))
@pytest.mark.parametrize("water_mass", (-si.ng, -si.ug, -si.mg, si.mg))
@pytest.mark.parametrize("RHi", (1.1, 1.0, 0.9))
@pytest.mark.parametrize("fastmath", (True, False))
@pytest.mark.parametrize("diffusion_coordinate", ("WaterMass", "WaterMassLogarithm"))
def test_iwc_differs_after_one_timestep(
    *, dt, water_mass, RHi, fastmath, diffusion_coordinate, dv=1 * si.m**3
):
    # arrange
    builder = Builder(
        n_sd=1,
        environment=MoistBox(dt=dt, dv=dv),
        backend=CPU(
            formulae=Formulae(
                fastmath=fastmath,
                particle_shape_and_density="MixedPhaseSpheres",
                diffusion_coordinate=diffusion_coordinate,
            )
        ),
    )
    builder.add_dynamic(AmbientThermodynamics())
    builder.add_dynamic(VapourDepositionOnIce())
    particulator = builder.build(
        attributes={
            "multiplicity": np.full(shape=(builder.particulator.n_sd,), fill_value=1e4),
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
    thd0 = particulator.formulae.state_variable_triplet.th_dry(
        th_std=particulator.formulae.trivia.th_std(
            p=particulator.environment["p"][0], T=particulator.environment["T"][0]
        ),
        water_vapour_mixing_ratio=rv0,
    )
    particulator.environment["thd"] = thd0

    # act
    iwc_old = particulator.products["ice water content"].get().copy()
    particulator.run(steps=1)

    # assert
    if water_mass < 0 and RHi != 1:
        if RHi > 1:
            assert particulator.environment["water_vapour_mixing_ratio"][0] < rv0
            assert particulator.environment["thd"][0] > thd0
            assert (particulator.products["ice water content"].get() > iwc_old).all()
        elif RHi < 1:
            assert particulator.environment["water_vapour_mixing_ratio"][0] > rv0
            assert particulator.environment["thd"][0] < thd0
            assert (particulator.products["ice water content"].get() < iwc_old).all()
    else:
        assert (particulator.products["ice water content"].get() == iwc_old).all()
        assert particulator.environment["water_vapour_mixing_ratio"][0] == rv0
        assert particulator.environment["thd"][0] == thd0
