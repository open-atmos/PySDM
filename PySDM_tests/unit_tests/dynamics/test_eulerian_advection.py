"""
Created at 14.07.2020
"""

from PySDM.dynamics.eulerian_advection import EulerianAdvection
from PySDM_tests.unit_tests.dummy_core import DummyCore
from PySDM_tests.unit_tests.dummy_environment import DummyEnvironment
import numpy as np


class TestEulerianAdvection:

    @staticmethod
    def test_update():
        # Arrange
        core = DummyCore()
        halo = 3
        grid = (11, 13)
        env = DummyEnvironment(grid=grid, halo=halo)
        env.register(core)
        env.qv[:] = 7.3
        env.thd[:] = 59.5
        env.pred['qv'][:] = 3.7
        env.pred['thd'][:] = 5.59
        core.environment = env

        sut = EulerianAdvection()
        sut.register(core)

        # Act
        sut()

        # Assert
        np.testing.assert_array_equal(env.get_qv(), env.get_predicted('qv').to_ndarray().reshape(grid))
        np.testing.assert_array_equal(env.get_thd(), env.get_predicted('thd').to_ndarray().reshape(grid))
        assert env.step_counter == 1
