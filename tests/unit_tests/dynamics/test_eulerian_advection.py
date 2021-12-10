# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
from PySDM.dynamics import EulerianAdvection
from ..dummy_particulator import DummyParticulator
from ..dummy_environment import DummyEnvironment
from ...backends_fixture import backend_class

assert hasattr(backend_class, '_pytestfixturefunction')


class TestEulerianAdvection:

    @staticmethod
    # pylint: disable=redefined-outer-name
    def test_update(backend_class):
        # Arrange
        particulator = DummyParticulator(backend_class)
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
        np.testing.assert_array_equal(
            env.get_qv(),
            env.get_predicted('qv').to_ndarray().reshape(grid))
        np.testing.assert_array_equal(
            env.get_thd(),
            env.get_predicted('thd').to_ndarray().reshape(grid))
