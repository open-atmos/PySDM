"""
Created at 2019
"""

from PySDM_examples.Arabas_and_Shima_2017_Fig_5.simulation import Simulation
from PySDM_examples.Arabas_and_Shima_2017_Fig_5.settings import setups
from PySDM.physics import formulae as phys
from PySDM.physics import constants as const
import pytest
import numpy as np


@pytest.mark.parametrize("settings_idx", range(len(setups)))
def test_initialisation(settings_idx):
    # Arrange
    settings = setups[settings_idx]

    pv0 = settings.p0 / (1 + const.eps / settings.q0)
    pd0 = settings.p0 - pv0
    rhod0 = pd0 / const.Rd / settings.T0
    thd0 = phys.th_std(pd0, settings.T0)

    # Act
    simulation = Simulation(settings)

    # Assert
    env = simulation.core.environment
    np.testing.assert_approx_equal(env['T'].to_ndarray(), settings.T0)
    np.testing.assert_approx_equal(env['RH'].to_ndarray(), pv0 / phys.pvs(settings.T0))
    np.testing.assert_approx_equal(env['p'].to_ndarray(), settings.p0)
    np.testing.assert_approx_equal(env['qv'].to_ndarray(), settings.q0)
    np.testing.assert_approx_equal(env['rhod'].to_ndarray(), rhod0)
    np.testing.assert_approx_equal(env['thd'].to_ndarray(), thd0)
