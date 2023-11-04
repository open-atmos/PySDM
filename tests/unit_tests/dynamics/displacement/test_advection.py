# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
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

    @staticmethod
    def test_advection(backend_class):
        # Arrange
        settings = DisplacementSettings()
        settings.grid = (3, 3)
        settings.courant_field_data = (np.ones((4, 3)), np.zeros((3, 4)))
        settings.positions = [[1.5], [1.5]]
        sut, particulator = settings.get_displacement(
            backend_class, scheme="ImplicitInSpace"
        )

        # Act
        sut()

        # Assert
        dim_x = 0
        np.testing.assert_array_equal(
            particulator.attributes["cell origin"][:, dim_x], np.array([2, 1])
        )

    @staticmethod
    def test_calculate_displacement(backend_class):
        # Arrange
        settings = DisplacementSettings()
        value_a = 0.1
        value_b = 0.2
        weight = 0.25
        settings.courant_field_data = (
            np.array([[value_a, value_b]]).T,
            np.array([[0, 0]]),
        )
        settings.positions = [[weight], [0]]
        sut, particulator = settings.get_displacement(
            backend_class, scheme="ExplicitInSpace", adaptive=False
        )

        # Act
        sut.calculate_displacement(
            sut.displacement,
            sut.courant,
            particulator.attributes["cell origin"],
            particulator.attributes["position in cell"],
        )

        # Assert
        np.testing.assert_equal(
            sut.displacement[0, slice(0, 1)].to_ndarray(),
            (1 - weight) * value_a + weight * value_b,
        )

    @staticmethod
    def test_calculate_displacement_dim1(backend_class):
        # Arrange
        settings = DisplacementSettings()
        value_a = 0.1
        value_b = 0.2
        weight = 0.25
        settings.courant_field_data = (
            np.array([[0, 0]]).T,
            np.array([[value_a, value_b]]),
        )
        settings.positions = [[0], [weight]]
        sut, particulator = settings.get_displacement(
            backend_class, scheme="ExplicitInSpace", adaptive=False
        )

        # Act
        sut.calculate_displacement(
            sut.displacement,
            sut.courant,
            particulator.attributes["cell origin"],
            particulator.attributes["position in cell"],
        )

        # Assert
        np.testing.assert_equal(
            sut.displacement[1, slice(0, 1)].to_ndarray(),
            (1 - weight) * value_a + weight * value_b,
        )

    @staticmethod
    def test_update_position(backend_class):
        # Arrange
        settings = DisplacementSettings()
        p_x = 0.1
        p_y = 0.2
        settings.positions = [[p_x], [p_y]]
        sut, particulator = settings.get_displacement(
            backend_class, scheme="ImplicitInSpace"
        )

        droplet_id = slice(0, 1)
        sut.displacement[:] = particulator.backend.Storage.from_ndarray(
            np.asarray(
                [
                    [
                        0.1,
                    ],
                    [0.2],
                ]
            )
        )

        # Act
        sut.update_position(
            particulator.attributes["position in cell"], sut.displacement
        )

        # Assert
        for d in range(2):
            np.testing.assert_array_almost_equal(
                particulator.attributes["position in cell"][d, droplet_id].to_ndarray(),
                settings.positions[d][droplet_id]
                + sut.displacement[d, droplet_id].to_ndarray(),
            )

    @staticmethod
    def test_update_cell_origin(backend_class):
        # Arrange
        settings = DisplacementSettings()
        sut, particulator = settings.get_displacement(
            backend_class, scheme="ImplicitInSpace"
        )

        droplet_id = 0
        state = particulator.attributes
        state["position in cell"][:] = particulator.backend.Storage.from_ndarray(
            np.asarray([[1.1], [1.2]])
        )

        # Act
        sut.update_cell_origin(state["cell origin"], state["position in cell"])

        # Assert
        for d in range(2):
            assert (
                state["cell origin"][d, droplet_id]
                == settings.positions[d][droplet_id] + 1
            )
            assert state["position in cell"][d, droplet_id] == (
                state["position in cell"][d, droplet_id]
                - np.floor(state["position in cell"][d, droplet_id])
            )

    @staticmethod
    def test_boundary_condition(backend_class):
        # Arrange
        settings = DisplacementSettings()
        sut, particulator = settings.get_displacement(
            backend_class, scheme="ImplicitInSpace"
        )

        droplet_id = 0
        state = particulator.attributes
        state["cell origin"][:] = particulator.backend.Storage.from_ndarray(
            np.asarray([[1], [1]])
        )

        # Act
        sut.boundary_condition(state["cell origin"])

        # Assert
        assert state["cell origin"][0, droplet_id] == 0
        assert state["cell origin"][1, droplet_id] == 0
