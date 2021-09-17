from PySDM.dynamics import EulerianAdvection
from ..dummy_particulator import DummyParticulator
from ..dummy_environment import DummyEnvironment
import numpy as np
# noinspection PyUnresolvedReferences
from ...backends_fixture import backend


class TestEulerianAdvection:

    @staticmethod
    def test_update(backend):
        # Arrange
        particulator = DummyParticulator(backend)
        halo = 3
        grid = (11, 13)
        env = DummyEnvironment(grid=grid, halo=halo)
        env.register(particulator)
        env.qv[:] = 7.3
        env.thd[:] = 59.5
        env.pred['qv'][:] = 3.7
        env.pred['thd'][:] = 5.59
        particulator.environment = env

        sut = EulerianAdvection(lambda: None)
        sut.register(particulator)

        # Act
        sut()

        # Assert
        np.testing.assert_array_equal(env.get_qv(), env.get_predicted('qv').to_ndarray().reshape(grid))
        np.testing.assert_array_equal(env.get_thd(), env.get_predicted('thd').to_ndarray().reshape(grid))
