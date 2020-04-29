"""
Created at 23.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import numpy as np
from .advection_test_setup import Setup


class TestExplicitEulerWithInterpolation:

    def test_single_cell(self):
        # Arrange
        setup = Setup()
        setup.courant_field_data = (np.array([[.1, .2]]).T, np.array([[.3, .4]]))
        setup.positions = [[0.5, 0.5]]
        sut, _ = setup.get_displacement()

        # Act
        sut()

        # Assert
        # TODO

    def test_advection(self):
        # Arrange
        setup = Setup()
        setup.grid = (3, 3)
        setup.courant_field_data = (np.ones((4, 3)), np.zeros((3, 4)))
        setup.positions = [[1.5, 1.5]]
        sut, particles = setup.get_displacement()

        # Act
        sut()

        # Assert
        np.testing.assert_array_equal(particles.state.cell_origin[0, :], np.array([2, 1]))

    def test_calculate_displacement(self):
        # Arrange
        setup = Setup()
        a = .1
        b = .2
        w = .25
        setup.courant_field_data = (np.array([[a, b]]).T, np.array([[0, 0]]))
        setup.positions = [[w, 0]]
        setup.scheme = 'FTFS'
        sut, particles = setup.get_displacement()

        # Act
        sut.calculate_displacement(sut.displacement, sut.courant,
                                   particles.state.cell_origin, particles.state.position_in_cell)

        # Assert
        np.testing.assert_equal(sut.displacement[0, 0], (1 - w) * a + w * b)

    def test_calculate_displacement_dim1(self):
        # Arrange
        setup = Setup()
        a = .1
        b = .2
        w = .25
        setup.courant_field_data = (np.array([[0, 0]]).T, np.array([[a, b]]))
        setup.positions = [[0, w]]
        setup.scheme = 'FTFS'
        sut, particles = setup.get_displacement()

        # Act
        sut.calculate_displacement(sut.displacement, sut.courant,
                                   particles.state.cell_origin, particles.state.position_in_cell)

        # Assert
        np.testing.assert_equal(sut.displacement[0, 1], (1 - w) * a + w * b)

    def test_update_position(self):
        # Arrange
        setup = Setup()
        px = .1
        py = .2
        setup.positions = [[px, py]]
        sut, particles = setup.get_displacement()

        droplet_id = 0
        sut.displacement[droplet_id, 0] = .1
        sut.displacement[droplet_id, 1] = .2

        # Act
        sut.update_position(particles.state.position_in_cell, sut.displacement)

        # Assert
        for d in range(2):
            assert particles.state.position_in_cell[droplet_id, d] == (
                   setup.positions[droplet_id][d] + sut.displacement[droplet_id, d]
            )

    def test_update_cell_origin(self):
        # Arrange
        setup = Setup()
        sut, particles = setup.get_displacement()

        droplet_id = 0
        state = particles.state
        state.position_in_cell[droplet_id, 0] = 1.1
        state.position_in_cell[droplet_id, 1] = 1.2

        # Act
        sut.update_cell_origin(state.cell_origin, state.position_in_cell)

        # Assert
        for d in range(2):
            assert state.cell_origin[droplet_id, d] == setup.positions[droplet_id][d] + 1
            assert state.position_in_cell[droplet_id, d] == (state.position_in_cell[droplet_id, d]
                                                             - np.floor(state.position_in_cell[droplet_id, d]))

    def test_boundary_condition(self):
        # Arrange
        setup = Setup()
        sut, particles = setup.get_displacement()

        droplet_id = 0
        state = particles.state
        state.cell_origin[droplet_id, 0] = 1.1
        state.cell_origin[droplet_id, 1] = 1.2

        # Act
        sut.boundary_condition(state.cell_origin)

        # Assert
        assert state.cell_origin[droplet_id, 0] == 0
        assert state.cell_origin[droplet_id, 1] == 0
