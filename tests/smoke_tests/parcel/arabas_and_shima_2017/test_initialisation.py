# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from PySDM_examples.Arabas_and_Shima_2017.settings import setups
from PySDM_examples.Arabas_and_Shima_2017.simulation import Simulation

from PySDM import Formulae

CONST = Formulae().constants


class TestInitialisation:
    @staticmethod
    def simulation_test(var, expected, setup):
        simulation = Simulation(setup)
        np.testing.assert_approx_equal(
            simulation.particulator.environment[var].to_ndarray(), expected
        )

    @staticmethod
    @pytest.mark.parametrize("settings_idx", range(len(setups)))
    def test_T_initialisation(settings_idx):
        setup = setups[settings_idx]
        TestInitialisation.simulation_test("T", setup.T0, setup)

    @staticmethod
    @pytest.mark.parametrize("settings_idx", range(len(setups)))
    def test_RH_initialisation(settings_idx):
        setup = setups[settings_idx]
        pv0 = setup.p0 / (1 + CONST.eps / setup.initial_water_vapour_mixing_ratio)
        pvs = setup.formulae.saturation_vapour_pressure.pvs_Celsius(setup.T0 - CONST.T0)
        TestInitialisation.simulation_test("RH", pv0 / pvs, setup)

    @staticmethod
    @pytest.mark.parametrize("settings_idx", range(len(setups)))
    def test_p_initialisation(settings_idx):
        setup = setups[settings_idx]
        TestInitialisation.simulation_test("p", setup.p0, setup)

    @staticmethod
    @pytest.mark.parametrize("settings_idx", range(len(setups)))
    def test_water_vapour_mixing_ratio_initialisation(settings_idx):
        setup = setups[settings_idx]
        TestInitialisation.simulation_test(
            "water_vapour_mixing_ratio", setup.initial_water_vapour_mixing_ratio, setup
        )

    @staticmethod
    @pytest.mark.parametrize("settings_idx", range(len(setups)))
    def test_rhod_initialisation(settings_idx):
        setup = setups[settings_idx]
        pv0 = setup.p0 / (1 + CONST.eps / setup.initial_water_vapour_mixing_ratio)
        pd0 = setup.p0 - pv0
        rhod0 = pd0 / CONST.Rd / setup.T0
        TestInitialisation.simulation_test("rhod", rhod0, setup)

    @staticmethod
    @pytest.mark.parametrize("settings_idx", range(len(setups)))
    def test_thd_initialisation(settings_idx):
        setup = setups[settings_idx]
        pv0 = setup.p0 / (1 + CONST.eps / setup.initial_water_vapour_mixing_ratio)
        pd0 = setup.p0 - pv0
        phys = Formulae().trivia
        thd0 = phys.th_std(pd0, setup.T0)
        TestInitialisation.simulation_test("thd", thd0, setup)
