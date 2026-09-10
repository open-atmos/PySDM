# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np

from PySDM.dynamics import EulerianAdvection

from ..dummy_particulator import DummyParticulator


class TestEulerianAdvection:  # pylint: disable=too-few-public-methods
    @staticmethod
    def test_update(backend_class):
        # Arrange
        grid = (11, 13)
        particulator = DummyParticulator(
            backend_class,
            n_sd=1,
            halo=3,
            grid=grid,
            attributes={
                "cell id": np.zeros(1, dtype=np.int64),
                "multiplicity": np.zeros(1, dtype=np.int64),
                "water mass": np.zeros(1, dtype=np.float64),
            },
        )
        env = particulator.environment
        env.water_vapour_mixing_ratio[:] = 7.3
        env.thd[:] = 59.5
        env.pred["water_vapour_mixing_ratio"][:] = 3.7
        env.pred["thd"][:] = 5.59
        particulator.dynamics["Displacement"] = None

        sut = EulerianAdvection(lambda _: None)
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
