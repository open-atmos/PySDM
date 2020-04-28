from PySDM_examples.Arabas_and_Shima_2017_Fig_5.simulation import Simulation
from PySDM_examples.Arabas_and_Shima_2017_Fig_5.setup import setups
from PySDM.simulation.physics import formulae as phys
from PySDM.simulation.physics import constants as const
import pytest
import numpy as np


@pytest.mark.parametrize("setup_idx", range(len(setups)))
def test_initialisation(setup_idx):
    # Arrange
    setup = setups[setup_idx]

    pv0 = setup.p0 / (1 + const.eps / setup.q0)
    pd0 = setup.p0 - pv0
    rhod0 = pd0 / const.Rd / setup.T0
    thd0 = phys.th_std(pd0, setup.T0)

    # Act
    simulation = Simulation(setup)

    # Assert
    env = simulation.particles.environment
    np.testing.assert_approx_equal(env['T'][:], setup.T0)
    np.testing.assert_approx_equal(env['RH'][:], pv0 / phys.pvs(setup.T0))
    np.testing.assert_approx_equal(env['p'][:], setup.p0)
    np.testing.assert_approx_equal(env['qv'][:], setup.q0)
    np.testing.assert_approx_equal(env['rhod'][:], rhod0)
    np.testing.assert_approx_equal(env['thd'][:], thd0)
