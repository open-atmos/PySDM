"""basic water vapor deposition on ice test"""

from typing import Iterable

import numpy as np
from matplotlib import pyplot
import pytest
import numba

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
DIFFUSION_ICE_CAPACITIES = ("Spherical", "Columnar")
COMMON = {
    "products": (IceWaterContent(),),
}


def make_particulator(
    *,
    dt: float,
    diffusion_coordinate: str,
    diffusion_ice_capacity: str,
    signed_water_masses: Iterable,
    temperature: float,
    pressure: float,
    RH_ice: float = None,
    RH_water: float = None,
    adaptive: bool = False,
    multiplicity: int = int(1e8),
):
    """instantiates a particulator with minimal components for testing ice depositional growth"""
    assert RH_water is None or RH_ice is None
    builder = Builder(
        n_sd=len(signed_water_masses),
        environment=MoistBox(dt=dt, dv=1 * si.m**3),
        backend=CPU(
            override_jit_flags={"parallel": False},
            formulae=Formulae(
                particle_shape_and_density="MixedPhaseSpheres",
                diffusion_coordinate=diffusion_coordinate,
                diffusion_ice_capacity=diffusion_ice_capacity,
            ),
        ),
    )
    builder.add_dynamic(AmbientThermodynamics())
    builder.add_dynamic(VapourDepositionOnIce(adaptive=adaptive))
    particulator = builder.build(
        attributes={
            "multiplicity": np.full(
                shape=(builder.particulator.n_sd,), fill_value=multiplicity
            ),
            "signed water mass": np.asarray(signed_water_masses),
        },
        products=(IceWaterContent(),),
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
    particulator.environment["water_vapour_mixing_ratio"] = (
        particulator.formulae.constants.eps
        * vapour_pressure
        / (particulator.environment["p"][0] - vapour_pressure)
    )
    particulator.environment["rhod"] = (
        particulator.environment["p"][0] - vapour_pressure
    ) / (particulator.environment["T"][0] * particulator.formulae.constants.Rd)
    particulator.environment["thd"] = (
        particulator.formulae.state_variable_triplet.th_dry(
            th_std=particulator.formulae.trivia.th_std(
                p=particulator.environment["p"][0], T=particulator.environment["T"][0]
            ),
            water_vapour_mixing_ratio=particulator.environment[
                "water_vapour_mixing_ratio"
            ][0],
        )
    )
    return particulator


class TestVapourDepositionOnIce:
    """groups tests for ice depositional growth"""

    @staticmethod
    @pytest.mark.parametrize("water_mass", (-si.ng, -si.mg, si.mg))
    @pytest.mark.parametrize("RHi", (1.1, 1.0, 0.9))
    @pytest.mark.parametrize("diffusion_coordinate", DIFFUSION_COORDINATES)
    @pytest.mark.parametrize("diffusion_ice_capacity", DIFFUSION_ICE_CAPACITIES)
    def test_iwc_differs_after_one_timestep(
        *, water_mass, RHi, diffusion_coordinate, diffusion_ice_capacity
    ):
        """sanity checks for sign of changes in IWC and ambient thermodynamics"""
        # arrange
        particulator = make_particulator(
            temperature=250 * si.K,
            pressure=500 * si.hPa,
            diffusion_coordinate=diffusion_coordinate,
            diffusion_ice_capacity=diffusion_ice_capacity,
            signed_water_masses=[water_mass],
            RH_ice=RHi,
            dt=0.1 * si.s,
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
            np.testing.assert_approx_equal(
                particulator.products["ice water content"].get()[0], iwc_old
            )
            np.testing.assert_approx_equal(
                particulator.environment["water_vapour_mixing_ratio"][0], rv0
            )
            np.testing.assert_approx_equal(particulator.environment["thd"][0], thd0)

    @staticmethod
    @pytest.mark.parametrize(
        "dt, adaptive",
        (
            (0.01 * si.s, False),
            pytest.param(1.0 * si.s, False, marks=pytest.mark.xfail(strict=True)),
        ),
    )
    @pytest.mark.parametrize("diffusion_coordinate", DIFFUSION_COORDINATES)
    def test_growth_rates_against_spichtinger_and_gierens_2009_fig_5(
        diffusion_coordinate, dt, adaptive, plot=False
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
                diffusion_ice_capacity="Columnar",
                signed_water_masses=-initial_water_masses,
                RH_water=1,
                temperature=temperature,
                dt=dt,
                adaptive=adaptive,
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

        assert 1.8e-15 * si.kg / si.s < dm_dt[230 * si.K][0] < 4.0e-15 * si.kg / si.s
        assert 7.0e-17 * si.kg / si.s < dm_dt[200 * si.K][0] < 1.0e-16 * si.kg / si.s
        assert 1.3e-12 * si.kg / si.s < dm_dt[230 * si.K][-1] < 1.5e-12 * si.kg / si.s
        assert 6.0e-14 * si.kg / si.s < dm_dt[200 * si.K][-1] < 1.2e-13 * si.kg / si.s

    @staticmethod
    @pytest.mark.parametrize("diffusion_coordinate", DIFFUSION_COORDINATES)
    @pytest.mark.parametrize("diffusion_ice_capacity", DIFFUSION_ICE_CAPACITIES)
    def test_relative_mass_rates(*, diffusion_coordinate, diffusion_ice_capacity):
        # arrange
        water_mass_init = np.logspace(-16, -6, num=11) * si.kg
        particulator = make_particulator(
            temperature=250 * si.K,
            pressure=500 * si.hPa,
            RH_ice=1.1,
            signed_water_masses=-water_mass_init,
            diffusion_coordinate=diffusion_coordinate,
            diffusion_ice_capacity=diffusion_ice_capacity,
            dt=0.1 * si.s,
        )

        # act
        particulator.run(steps=1)

        # assert
        water_mass_new = particulator.attributes["water mass"].to_ndarray()
        relative_growth = (water_mass_new - water_mass_init) / water_mass_init
        assert all(relative_growth > 0.0)
        assert all(np.diff(relative_growth) < 0.0)

    @staticmethod
    @pytest.mark.parametrize(
        "rh_ice, multiplicity, diffusion_coordinate",
        (
            (1.5, 1e8, "WaterMass"),
            (1.0, 1e8, "WaterMass"),
            pytest.param(
                0.5,
                1e8,
                "WaterMass",
                marks=pytest.mark.xfail(
                    strict=numba.config.DISABLE_JIT  # pylint:disable=no-member
                ),
            ),
            (1.5, 1, "WaterMass"),
            (1.0, 1, "WaterMass"),
            (0.5, 1, "WaterMass"),
            (1.5, 1e8, "WaterMassLogarithm"),
            (1.0, 1e8, "WaterMassLogarithm"),
            pytest.param(
                0.5,
                1e8,
                "WaterMassLogarithm",
                marks=pytest.mark.xfail(
                    strict=numba.config.DISABLE_JIT  # pylint:disable=no-member
                ),
            ),
            pytest.param(
                1.5, 1, "WaterMassLogarithm", marks=pytest.mark.xfail(strict=True)
            ),
            (1.0, 1, "WaterMassLogarithm"),
            (0.5, 1, "WaterMassLogarithm"),
        ),
    )
    @pytest.mark.parametrize("diffusion_ice_capacity", DIFFUSION_ICE_CAPACITIES)
    def test_mass_conservation_under_adaptivity(
        rh_ice,
        diffusion_ice_capacity,
        diffusion_coordinate,
        multiplicity,
    ):
        # arrange
        water_mass_init = np.logspace(-15, -6, num=11) * si.kg
        particulator = make_particulator(
            adaptive=True,
            dt=10 * si.s,
            diffusion_coordinate=diffusion_coordinate,
            diffusion_ice_capacity=diffusion_ice_capacity,
            signed_water_masses=-water_mass_init,
            temperature=250 * si.K,
            pressure=800 * si.hPa,
            RH_ice=rh_ice,
            multiplicity=multiplicity,
        )

        def total_water_mass_in_the_system(attr, env):
            return np.dot(
                attr["water mass"].to_ndarray(),
                attr["multiplicity"].to_ndarray(),
            ) + (env["water_vapour_mixing_ratio"][0] * env["rhod"][0] * env.mesh.dv)

        # act
        m0 = total_water_mass_in_the_system(
            particulator.attributes, particulator.environment
        )
        particulator.run(steps=1)
        m1 = total_water_mass_in_the_system(
            particulator.attributes, particulator.environment
        )

        # assert
        np.testing.assert_almost_equal(m0, m1)


# TODO #1524: test is updraft matters
# TODO #1524: test if order of condensation/deposition matters
