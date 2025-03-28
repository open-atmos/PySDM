"""basic water vapor deposition on ice test"""

import numpy as np
from matplotlib import pyplot

import pytest

from PySDM.physics import si, in_unit

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
    """env providing get_predicted() logic from Moist with box-model basics"""

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
    "environment": MoistBox(dt=0.01 * si.s, dv=1 * si.m**3),
    "products": (IceWaterContent(),),
    "formulae": {
        f"{diffusion_coordinate}": Formulae(
            particle_shape_and_density="MixedPhaseSpheres",
            diffusion_coordinate=diffusion_coordinate,
        )
        for diffusion_coordinate in DIFFUSION_COORDINATES
    },
}


def make_particulator(
    *,
    diffusion_coordinate,
    signed_water_masses,
    temperature,
    pressure,
    RH_ice=None,
    RH_water=None,
):
    """instantiates a particulator with minimal components for testing ice depositional growth"""
    assert RH_water is None or RH_ice is None
    builder = Builder(
        n_sd=len(signed_water_masses),
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
            "signed water mass": np.asarray(signed_water_masses),
        },
        products=COMMON["products"],
    )
    particulator.environment["T"] = temperature
    particulator.environment["p"] = pressure
    pvs_ice = particulator.formulae.saturation_vapour_pressure.pvs_ice(
        particulator.environment["T"][0]
    )
    pvs_water = particulator.formulae.saturation_vapour_pressure.pvs_water(
        particulator.environment["T"][0]
    )
    vapour_pressure = RH_ice * pvs_ice if RH_ice else RH_water * pvs_water
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
    return particulator


class TestVapourDepositionOnIce:
    """groups tests for ice depositional growth"""

    @staticmethod
    @pytest.mark.parametrize("water_mass", (-si.ng, -si.mg, si.mg))
    @pytest.mark.parametrize("RHi", (1.1, 1.0, 0.9))
    @pytest.mark.parametrize("diffusion_coordinate", DIFFUSION_COORDINATES)
    def test_iwc_differs_after_one_timestep(*, water_mass, RHi, diffusion_coordinate):
        """sanity checks for sign of changes in IWC and ambient thermodynamics"""
        # arrange
        particulator = make_particulator(
            temperature=250 * si.K,
            pressure=500 * si.hPa,
            diffusion_coordinate=diffusion_coordinate,
            signed_water_masses=[water_mass],
            RH_ice=RHi,
        )
        rv0 = particulator.environment["water_vapour_mixing_ratio"][0]
        thd0 = particulator.environment["thd"][0]

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

    @staticmethod
    @pytest.mark.parametrize("diffusion_coordinate", DIFFUSION_COORDINATES)
    def test_growth_rates_against_spichtinger_and_gierens_2009_fig_5(
        diffusion_coordinate, plot=False
    ):
        """Fig. 5 in [Spichtinger & Gierens 2009](https://doi.org/10.5194/acp-9-685-2009)"""
        # arrange
        initial_water_masses = (
            np.logspace(base=10, start=-16, stop=-8.5, num=10) * si.kg
        )
        dm_dt = {}

        # act
        for temperature in np.linspace(200, 230, endpoint=True, num=4) * si.K:
            particulator = make_particulator(
                pressure=300 * si.hPa,
                diffusion_coordinate=diffusion_coordinate,
                signed_water_masses=-initial_water_masses,
                RH_water=1,
                temperature=temperature,
            )
            particulator.run(steps=1)
            dm_dt[temperature] = (
                particulator.attributes["water mass"].to_ndarray()
                - initial_water_masses
            ) / particulator.environment.dt

        pyplot.xlabel("mass (kg)")
        pyplot.ylabel("mass rate (kg/s)")
        pyplot.xlim(initial_water_masses[0], initial_water_masses[-1])
        pyplot.xscale("log")
        pyplot.ylim(1e-16, 1e-11)
        pyplot.yscale("log")
        pyplot.grid()
        pyplot.title(f"p={in_unit(particulator.environment['p'][0], si.hPa)} hPa")
        for temperature, mass_rate in dm_dt.items():
            pyplot.plot(initial_water_masses, mass_rate, color="black")
            pyplot.annotate(
                text=f" T={temperature}K", xy=(initial_water_masses[-1], mass_rate[-1])
            )

        # plot
        if plot:
            pyplot.show()
        else:
            pyplot.clf()

        # assert
        assert (dm_dt[200 * si.K] < dm_dt[210 * si.K]).all()
        assert (dm_dt[210 * si.K] < dm_dt[220 * si.K]).all()
        assert (dm_dt[220 * si.K] < dm_dt[230 * si.K]).all()
        for mass_rate in dm_dt.values():
            assert (np.diff(mass_rate) > 0).all()
        assert 0.2e-14 * si.kg / si.s < dm_dt[230 * si.K][0] < 0.3e-14 * si.kg / si.s
        assert 0.8e-16 * si.kg / si.s < dm_dt[200 * si.K][0] < 0.9e-16 * si.kg / si.s
        assert 1.1e-12 * si.kg / si.s < dm_dt[230 * si.K][-1] < 1.2e-12 * si.kg / si.s
        assert 4.8e-14 * si.kg / si.s < dm_dt[200 * si.K][-1] < 4.9e-14 * si.kg / si.s

    @staticmethod
    @pytest.mark.parametrize("diffusion_coordinate", DIFFUSION_COORDINATES)
    def test_relative_mass_rates(*, diffusion_coordinate):
        # arrange
        water_mass_init = np.logspace(-16, -6, num=11) * si.kg
        particulator = make_particulator(
            temperature=250 * si.K,
            pressure=500 * si.hPa,
            RH_ice=1.1,
            signed_water_masses=-water_mass_init,
            diffusion_coordinate=diffusion_coordinate,
        )

        # act
        particulator.run(steps=1)

        # assert
        water_mass_new = particulator.attributes["water mass"].to_ndarray()
        relative_growth = (water_mass_new - water_mass_init) / water_mass_init
        assert all(relative_growth > 0.0)
        assert all(np.diff(relative_growth) < 0.0)
