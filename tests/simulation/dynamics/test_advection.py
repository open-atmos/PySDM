"""
Created at 23.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from SDM.backends.default import Default
from SDM.simulation.dynamics.advection import Advection
from SDM.simulation.state import State
import numpy as np


class TestExplicitEulerWithInterpolation:

    def test_single_cell(self):
        n = np.ones(1)
        n_sd = len(n)
        positions = Default.from_ndarray(np.array([[0.5, 0.5]]))
        courant_field = (np.array([[.1, .2]]).T, np.array([[.3, .4]]))
        state = State.state_2d(n=n, grid=(1, 1), intensive={}, extensive={}, positions=positions, backend=Default)
        sut = Advection(n_sd=n_sd, courant_field=courant_field, backend=Default)

        sut(state=state)

    def test_advection(self):
        n = np.ones(1)
        n_sd = len(n)
        positions = Default.from_ndarray(np.array([[1.5, 1.5]]))
        courant_field = (np.ones((4, 3)), np.zeros((3, 4)))
        state = State.state_2d(n=n, grid=(3, 3), intensive={}, extensive={}, backend=Default, positions=positions)
        sut = Advection(n_sd=n_sd, courant_field=courant_field, backend=Default)

        sut(state=state)

        np.testing.assert_array_equal(state.cell_origin[0, :], np.array([2, 1]))

    def test_calculate_displacement(self):
        # Arrange
        n = np.ones(1)
        n_sd = len(n)
        a = .1
        b = .2
        w = .25
        positions = Default.from_ndarray(np.array([[w, 0]]))
        courant_field = (np.array([[a, b]]).T, np.array([[0, 0]]))
        state = State.state_2d(n=n, grid=(1, 1), intensive={}, extensive={}, backend=Default, positions=positions)
        sut = Advection(n_sd=n_sd, courant_field=courant_field, backend=Default)

        # Act
        sut.calculate_displacement(state)

        # Assert
        np.testing.assert_equal(sut.displacement[0, 0], w * a + (1 - w) * b)

    def test_update_position(self):
        # Arrange
        n = np.ones(1)
        n_sd = len(n)
        droplet_id = 0
        px = .1
        py = .2
        initial_position = Default.from_ndarray(np.array([[px, py]]))
        dummy_courant_field = (np.array([[0, 0]]).T, np.array([[0, 0]]))
        state = State.state_2d(n=n, grid=(1, 1), intensive={}, extensive={}, backend=Default, positions=initial_position)
        sut = Advection(n_sd=n_sd, courant_field=dummy_courant_field, backend=Default)
        sut.displacement[droplet_id, 0] = .1
        sut.displacement[droplet_id, 1] = .2

        # Act
        sut.update_position(state)

        # Assert
        for d in range(2):
            assert state.position_in_cell[droplet_id, d] == (
                    initial_position[droplet_id, d] + sut.displacement[droplet_id, d]
            )

    def test_update_cell_origin(self):
        # Arrange
        n = np.ones(1)
        n_sd = len(n)
        droplet_id = 0
        initial_position = Default.from_ndarray(np.array([[0, 0]]))
        dummy_courant_field = (np.array([[0, 0]]).T, np.array([[0, 0]]))
        state = State.state_2d(n=n, grid=(1, 1), intensive={}, extensive={}, backend=Default, positions=initial_position)
        sut = Advection(n_sd=n_sd, courant_field=dummy_courant_field, backend=Default)
        state.position_in_cell[droplet_id, 0] = 1.1
        state.position_in_cell[droplet_id, 1] = 1.2

        # Act
        sut.update_cell_origin(state)

        # Assert
        for d in range(2):
            assert state.cell_origin[droplet_id, d] == initial_position[droplet_id, d] + 1
            assert state.position_in_cell[droplet_id, d] == state.position_in_cell[droplet_id, d] - np.floor(state.position_in_cell[droplet_id, d])

    def test_boundary_condition(self):
        # Arrange
        n = np.ones(1)
        n_sd = len(n)
        droplet_id = 0
        initial_position = Default.from_ndarray(np.array([[0, 0]]))
        dummy_courant_field = (np.array([[0, 0]]).T, np.array([[0, 0]]))
        state = State.state_2d(n=n, grid=(1, 1), intensive={}, extensive={}, backend=Default, positions=initial_position)
        sut = Advection(n_sd=n_sd, courant_field=dummy_courant_field, backend=Default)
        state.cell_origin[droplet_id, 0] = 1.1
        state.cell_origin[droplet_id, 1] = 1.2

        # Act
        sut.boundary_condition(state)

        # Assert
        assert state.cell_origin[droplet_id, 0] == 0
        assert state.cell_origin[droplet_id, 1] == 0

    def test_recalculate_cell_id(self):
        # Arrange
        n = np.ones(1)
        n_sd = len(n)
        droplet_id = 0
        initial_position = Default.from_ndarray(np.array([[0, 0]]))
        dummy_courant_field = (np.array([[0, 0]]).T, np.array([[0, 0]]))
        state = State.state_2d(n=n, grid=(1, 1), intensive={}, extensive={}, backend=Default, positions=initial_position)
        sut = Advection(n_sd=n_sd, courant_field=dummy_courant_field, backend=Default)
        state.cell_origin[droplet_id, 0] = .1
        state.cell_origin[droplet_id, 1] = .2
        state.cell_id[droplet_id] = -1

        # Act
        sut.recalculate_cell_id(state)

        # Assert
        assert state.cell_id[droplet_id] == 0
