# pylint: disable=missing-module-docstring

from collections import namedtuple
from functools import partial
from contextlib import contextmanager
import pytest
import numpy as np
from scipy.optimize import fsolve

from PySDM_examples.Loftus_and_Wordsworth_2021.planet import (
    EarthLike,
    Earth,
    EarlyMars,
    Jupiter,
    Saturn,
    K2_18B,
)
from PySDM_examples.Loftus_and_Wordsworth_2021.simulation import Simulation
from PySDM_examples.Loftus_and_Wordsworth_2021.parcel import AlienParcel
from PySDM_examples.Loftus_and_Wordsworth_2021 import Settings

from PySDM import Formulae
from PySDM.physics import si


class TestLoftusWordsworth2021:

    @contextmanager
    @staticmethod
    def _get_test_resources():
        formulae = Formulae(
            ventilation="PruppacherAndRasmussen1979",
            saturation_vapour_pressure="AugustRocheMagnus",
            diffusion_coordinate="WaterMassLogarithm",
        )
        earth_like = EarthLike()
        try:
            yield formulae, earth_like
        finally:
            pass

    def test_planet_classes(self):
        """Test planet class instantiation and basic properties."""
        planets = [EarthLike(), Earth(), EarlyMars(), Jupiter(), Saturn(), K2_18B()]

        for planet in planets:
            assert planet.g_std > 0
            assert planet.T_STP > 0
            assert planet.p_STP > 0
            assert planet.RH_zref >= 0
            assert planet.RH_zref <= 1

            # atmospheric composition sums to 1 or less
            total_conc = (
                planet.dry_molar_conc_H2
                + planet.dry_molar_conc_He
                + planet.dry_molar_conc_N2
                + planet.dry_molar_conc_O2
                + planet.dry_molar_conc_CO2
            )
            assert total_conc <= 1.01, (
                f"Total molar concentration {total_conc} "
                + f"exceeds 1.01 for {planet.__class__.__name__}"
            )

    def test_water_vapour_mixing_ratio_calculation(self):
        """Test water vapour mixing ratio calculation."""
        with TestLoftusWordsworth2021._get_test_resources() as (formulae, earth_like):
            const = formulae.constants
            planet = earth_like

            pvs = formulae.saturation_vapour_pressure.pvs_water(planet.T_STP)
            initial_water_vapour_mixing_ratio = const.eps / (
                planet.p_STP / planet.RH_zref / pvs - 1
            )

            assert initial_water_vapour_mixing_ratio > 0
            assert initial_water_vapour_mixing_ratio < 0.1  # Should be less than 10%

    def test_alien_parcel_initialization(self):
        """Test AlienParcel class initialization."""
        parcel = AlienParcel(
            dt=1.0 * si.second,
            mass_of_dry_air=1e5 * si.kg,
            pcloud=90000 * si.pascal,
            initial_water_vapour_mixing_ratio=0.01,
            Tcloud=280 * si.kelvin,
            w=0,
            zcloud=1000 * si.m,
        )
        assert parcel is not None

    # pylint: disable=too-many-arguments
    @pytest.mark.parametrize(
        "r_wet_val, mass_of_dry_air_val, iwvmr_val, pcloud_val, Zcloud_val, Tcloud_val",
        [
            (1e-4, 1e5, 0.01, 90000, 1000, 280),
            (1e-5, 1e4, 0.005, 80000, 500, 270),
            (2e-4, 2e5, 0.02, 95000, 1500, 290),
        ],
    )
    def test_simulation_class(
        self,
        r_wet_val,
        mass_of_dry_air_val,
        iwvmr_val,
        pcloud_val,
        Zcloud_val,
        Tcloud_val,
    ):
        """
        Test Simulation class initialization and basic functionality with parametrized settings.
        """
        with TestLoftusWordsworth2021._get_test_resources() as (formulae, earth_like):
            planet = earth_like

            settings = Settings(
                planet=planet,
                r_wet=r_wet_val * si.m,
                mass_of_dry_air=mass_of_dry_air_val * si.kg,
                initial_water_vapour_mixing_ratio=iwvmr_val,
                pcloud=pcloud_val * si.pascal,
                Zcloud=Zcloud_val * si.m,
                Tcloud=Tcloud_val * si.kelvin,
                formulae=formulae,
            )

            simulation = Simulation(settings)

            assert hasattr(simulation, "particulator")
            assert hasattr(simulation, "run")
            assert hasattr(simulation, "save")

            products = simulation.particulator.products
            required_products = ["radius_m1", "z", "RH", "t"]
            for product in required_products:
                assert product in products

    # pylint: disable=too-many-arguments
    @pytest.mark.parametrize(
        "r_wet_val, mass_of_dry_air_val, iwvmr_val, pcloud_val, Zcloud_val, Tcloud_val",
        [
            (1e-5, 1e5, 0.01, 90000, 100, 280),
            (1e-5, 1e4, 0.005, 80000, 500, 270),
            (2e-4, 2e5, 0.02, 95000, 1500, 290),
        ],
    )
    def test_simulation_run_basic(
        self,
        r_wet_val,
        mass_of_dry_air_val,
        iwvmr_val,
        pcloud_val,
        Zcloud_val,
        Tcloud_val,
    ):
        """Test basic simulation run functionality."""
        with TestLoftusWordsworth2021._get_test_resources() as (formulae, earth_like):
            planet = earth_like

            settings = Settings(
                planet=planet,
                r_wet=r_wet_val * si.m,
                mass_of_dry_air=mass_of_dry_air_val * si.kg,
                initial_water_vapour_mixing_ratio=iwvmr_val,
                pcloud=pcloud_val * si.pascal,
                Zcloud=Zcloud_val * si.m,
                Tcloud=Tcloud_val * si.kelvin,
                formulae=formulae,
            )

            simulation = Simulation(settings)
            output = simulation.run()

            assert "r" in output
            assert "S" in output
            assert "z" in output
            assert "t" in output

            assert output["r"] is not None
            assert output["S"] is not None
            assert output["z"] is not None
            assert output["t"] is not None

            assert len(output["r"]) > 0, "Output array 'r' is empty"
            assert len(output["S"]) > 0, "Output array 'S' is empty"
            assert len(output["z"]) > 0, "Output array 'z' is empty"
            assert len(output["t"]) > 0, "Output array 't' is empty"

            lengths = [len(one_output) for one_output in output.values()]
            assert all(
                length == lengths[0] for length in lengths
            ), "Not all output arrays have the same length"

    def test_saturation_at_cloud_base(self):
        formulae = Formulae(
            ventilation="PruppacherAndRasmussen1979",
            saturation_vapour_pressure="AugustRocheMagnus",
            diffusion_coordinate="WaterMassLogarithm",
        )

        new_Earth = EarthLike()

        RH_array = np.linspace(0.25, 0.99, 50)
        const = formulae.constants

        def mix(dry, vap, ratio):
            return (dry + ratio * vap) / (1 + ratio)

        def f(x, water_mixing_ratio, params):
            return water_mixing_ratio / (
                water_mixing_ratio + const.eps
            ) * params.p_stp * (x / params.t_stp) ** (
                params.c_p / params.Rair
            ) - formulae.saturation_vapour_pressure.pvs_water(
                x
            )

        for RH in RH_array[::-1]:
            new_Earth.RH_zref = RH

            initial_water_vapour_mixing_ratio = const.eps / (
                new_Earth.p_STP
                / new_Earth.RH_zref
                / formulae.saturation_vapour_pressure.pvs_water(new_Earth.T_STP)
                - 1
            )

            c_p = mix(const.c_pd, const.c_pv, initial_water_vapour_mixing_ratio)

            tdews = fsolve(
                partial(
                    f,
                    water_mixing_ratio=initial_water_vapour_mixing_ratio,
                    params=namedtuple(
                        "params",
                        ["p_stp", "t_stp", "c_p", "Rair"],
                    )(
                        p_stp=new_Earth.p_STP,
                        t_stp=new_Earth.T_STP,
                        c_p=c_p,
                        Rair=mix(const.Rd, const.Rv, initial_water_vapour_mixing_ratio),
                    ),
                ),
                [150, 300],
            )
            Tcloud = np.max(tdews)
            thstd = formulae.trivia.th_std(new_Earth.p_STP, new_Earth.T_STP)

            hydro = formulae.hydrostatics
            pcloud = (
                hydro.p_of_z_assuming_const_th_and_initial_water_vapour_mixing_ratio(
                    new_Earth.p_STP,
                    thstd,
                    initial_water_vapour_mixing_ratio,
                    (new_Earth.T_STP - Tcloud) * c_p / new_Earth.g_std,
                )
            )

            np.testing.assert_approx_equal(
                actual=pcloud
                * (
                    initial_water_vapour_mixing_ratio
                    / (initial_water_vapour_mixing_ratio + const.eps)
                )
                / formulae.saturation_vapour_pressure.pvs_water(Tcloud),
                desired=1,
                significant=4,
            )
