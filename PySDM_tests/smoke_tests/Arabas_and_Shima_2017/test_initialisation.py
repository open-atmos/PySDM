import numpy as np
import pytest
from PySDM.physics import Formulae, constants as const
from PySDM_examples.Arabas_and_Shima_2017.settings import setups
from PySDM_examples.Arabas_and_Shima_2017.simulation import Simulation


class TestInitialisation:

    @staticmethod
    def simulation_test(var, expected, setup):
        simulation = Simulation(setup)
        np.testing.assert_approx_equal(simulation.core.environment[var].to_ndarray(), expected)

    @staticmethod
    @pytest.mark.parametrize("settings_idx", range(len(setups)))
    def test_T_initialisation(settings_idx):
        setup = setups[settings_idx]
        TestInitialisation.simulation_test('T', setup.T0, setup)

    @staticmethod
    @pytest.mark.parametrize("settings_idx", range(len(setups)))
    def test_RH_initialisation(settings_idx):
        setup = setups[settings_idx]
        pv0 = setup.p0 / (1 + const.eps / setup.q0)
        pvs = setup.formulae.saturation_vapour_pressure.pvs_Celsius(setup.T0 - const.T0)
        TestInitialisation.simulation_test('RH', pv0 / pvs, setup)

    @staticmethod
    @pytest.mark.parametrize("settings_idx", range(len(setups)))
    def test_p_initialisation(settings_idx):
        setup = setups[settings_idx]
        TestInitialisation.simulation_test('p', setup.p0, setup)

    @staticmethod
    @pytest.mark.parametrize("settings_idx", range(len(setups)))
    def test_qv_initialisation(settings_idx):
        setup = setups[settings_idx]
        TestInitialisation.simulation_test('qv', setup.q0, setup)

    @staticmethod
    @pytest.mark.parametrize("settings_idx", range(len(setups)))
    def test_rhod_initialisation(settings_idx):
        setup = setups[settings_idx]
        pv0 = setup.p0 / (1 + const.eps / setup.q0)
        pd0 = setup.p0 - pv0
        rhod0 = pd0 / const.Rd / setup.T0
        TestInitialisation.simulation_test('rhod', rhod0, setup)

    @staticmethod
    @pytest.mark.parametrize("settings_idx", range(len(setups)))
    def test_thd_initialisation(settings_idx):
        setup = setups[settings_idx]
        pv0 = setup.p0 / (1 + const.eps / setup.q0)
        pd0 = setup.p0 - pv0
        phys = Formulae().trivia
        thd0 = phys.th_std(pd0, setup.T0)
        TestInitialisation.simulation_test('thd', thd0, setup)
