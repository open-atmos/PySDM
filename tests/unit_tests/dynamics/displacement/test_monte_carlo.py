import numpy as np
import pytest

from .displacement_settings import DisplacementSettings


class TestExplicitEulerWithInterpolation:
    @staticmethod
    @pytest.mark.parametrize(
        "positions, courant_field, expected_positions",
        (
            # 1D
            (
                [[0.5]],
                (np.array([0.1, 0.2]),),
                # (np.array([0.99, 0.99]),),
                [[0.65827668]],
            ),
            # 2D
            (
                [[0.5], [0.5]],
                (
                    np.array([0.1, 0.2]).reshape((2, 1)),
                    np.array([0.3, 0.4]).reshape((1, 2)),
                ),
                [[0.65827668], [0.86931224]],
            ),
            # 3D
            (
                [[0.5], [0.5], [0.5]],
                (
                    np.array([0.1, 0.2]).reshape((2, 1, 1)),
                    np.array([0.3, 0.4]).reshape((1, 2, 1)),
                    np.array([0.2, 0.3]).reshape((1, 1, 2)),
                ),
                [
                    [0.65827668],
                    [0.86931224],
                    [0.76379446],
                ],
            ),
        ),
    )



    @staticmethod
    def test_calculate_displacement(backend_class):
        # Arrange
        settings = DisplacementSettings()
        value_a = 0.1
        value_b = 0.2
        weight = 0.125
        settings.courant_field_data = (
            np.array([[value_a, value_b]]).T,
            np.array([[0, 0]]),
        )
        settings.positions = [[weight], [0]]
        sut, particulator = settings.get_displacement(
            backend_class, scheme="ExplicitInSpace", adaptive=False, enable_monte_carlo=True
        )
        # Act
        sut.calculate_displacement(
            displacement=sut.displacement,
            courant=sut.courant,
            cell_origin=particulator.attributes["cell origin"],
            position_in_cell=particulator.attributes["position in cell"],
            cell_id=particulator.attributes["cell id"],
        )

        # Assert
        if backend_class.__name__ == "ThrustRTC":
            np.testing.assert_equal(
                sut.displacement[0, slice(0, 1)].to_ndarray(),
                0.1125
            )

        if backend_class.__name__ == "Numba":
            np.testing.assert_equal(
                sut.displacement[0, slice(0, 1)].to_ndarray(),
                0.125
            )

    @staticmethod
    def test_calculate_displacement_2(backend_class):
        # Arrange
        settings = DisplacementSettings()
        value_a = 1
        value_b = 1
        weight = 0.125
        settings.courant_field_data = (
            np.array([[value_a, value_b]]).T,
            np.array([[0, 0]]),
        )
        settings.positions = [[weight], [0]]
        sut, particulator = settings.get_displacement(
            backend_class, scheme="ExplicitInSpace", adaptive=False, enable_monte_carlo=True
        )
        # Act
        sut.calculate_displacement(
            displacement=sut.displacement,
            courant=sut.courant,
            cell_origin=particulator.attributes["cell origin"],
            position_in_cell=particulator.attributes["position in cell"],
            cell_id=particulator.attributes["cell id"],
        )

        # Assert
        if backend_class.__name__ == "ThrustRTC":
            np.testing.assert_equal(
                sut.displacement[0, slice(0, 1)].to_ndarray(),
                1.
            )

        if backend_class.__name__ == "Numba":
            np.testing.assert_equal(
                sut.displacement[0, slice(0, 1)].to_ndarray(),
                2.125
            )
            
    def test_single_cell(
        backend_class, positions, expected_positions, courant_field: tuple
    ):
        # Arrange
        settings = DisplacementSettings(
            courant_field_data=courant_field,
            positions=positions,
            grid=tuple([1] * len(courant_field)),
            n_sd=len(positions[0]),
        )
        sut, particulator = settings.get_displacement(
            backend_class, scheme="ImplicitInSpace"
        )

        # Act
        sut()

        # Assert
        np.testing.assert_array_almost_equal(
            np.asarray(expected_positions),
            particulator.attributes["position in cell"].to_ndarray(),
        )
