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


DIFFUSION_COORDINATES = ("WaterMass", "WaterMassLogarithm")
COMMON = {
    "environment": MoistBox(dt=1 * si.s, dv=1 * si.m**3),
    "products": (IceWaterContent(),),
    "formulae": {
        f"{diffusion_coordinate}": Formulae(
            particle_shape_and_density="MixedPhaseSpheres",
            diffusion_coordinate=diffusion_coordinate,
        )
        for diffusion_coordinate in DIFFUSION_COORDINATES
    },
}


@pytest.mark.parametrize("water_mass", (-si.ng, -si.mg, si.mg))
@pytest.mark.parametrize("RHi", (1.1, 1.0, 0.9))
@pytest.mark.parametrize("diffusion_coordinate", DIFFUSION_COORDINATES)
def test_iwc_differs_after_one_timestep(*, water_mass, RHi, diffusion_coordinate):
    # arrange
    builder = Builder(
        n_sd=1,
        environment=COMMON["environment"],
        backend=CPU(
            formulae=COMMON["formulae"][diffusion_coordinate],
            override_jit_flags={"parallel": False},
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
        products=COMMON["products"],
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
    iwc_old = particulator.products["ice water content"].get()[0]
    particulator.run(steps=1)

    # assert
    if water_mass < 0 and RHi != 1:
        if RHi > 1:
            assert particulator.environment["water_vapour_mixing_ratio"][0] < rv0
            assert particulator.environment["thd"][0] > thd0
            assert particulator.products["ice water content"].get()[0] > iwc_old
        elif RHi < 1:
            assert particulator.environment["water_vapour_mixing_ratio"][0] > rv0
            assert particulator.environment["thd"][0] < thd0
            assert particulator.products["ice water content"].get()[0] < iwc_old
    else:
        assert particulator.products["ice water content"].get()[0] == iwc_old
        assert particulator.environment["water_vapour_mixing_ratio"][0] == rv0
        assert particulator.environment["thd"][0] == thd0
