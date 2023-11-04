# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np

from PySDM.dynamics import EulerianAdvection

from ..dummy_environment import DummyEnvironment
from ..dummy_particulator import DummyParticulator


class TestEulerianAdvection:  # pylint: disable=too-few-public-methods
    @staticmethod
    def test_update(backend_class):
        # Arrange
        particulator = DummyParticulator(backend_class)
        halo = 3
        grid = (11, 13)
        env = DummyEnvironment(grid=grid, halo=halo)
        env.register(particulator)
        env.water_vapour_mixing_ratio[:] = 7.3
        env.thd[:] = 59.5
        env.pred["water_vapour_mixing_ratio"][:] = 3.7
        env.pred["thd"][:] = 5.59
        particulator.environment = env

        sut = EulerianAdvection(lambda: None)
        sut.register(particulator)

        # Act
        sut()

        # Assert
        np.testing.assert_array_equal(
            env.get_water_vapour_mixing_ratio(),
            env.get_predicted("water_vapour_mixing_ratio").to_ndarray().reshape(grid),
        )
        np.testing.assert_array_equal(
            env.get_thd(), env.get_predicted("thd").to_ndarray().reshape(grid)
        )
