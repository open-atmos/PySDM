"""
Created at 23.10.2019
"""

import numpy as np
from .displacement_setup import Setup


class TestExplicitEulerWithInterpolation:

    def test_single_cell(self):
        # Arrange
        setup = Setup()
        setup.courant_field_data = (np.array([[.1, .2]]).T, np.array([[.3, .4]]))
        setup.positions = [[0.5], [0.5]]
        sut, _ = setup.get_displacement()

        # Act
        sut()

        # Assert
        pass

    def test_advection(self):
        # Arrange
        setup = Setup()
        setup.grid = (3, 3)
        setup.courant_field_data = (np.ones((4, 3)), np.zeros((3, 4)))
        setup.positions = [[1.5], [1.5]]
        sut, particles = setup.get_displacement()

        # Act
        sut()

        # Assert
        np.testing.assert_array_equal(particles.state['cell origin'][:, 0], np.array([2, 1]))

    def test_calculate_displacement(self):
        # Arrange
        setup = Setup()
        a = .1
        b = .2
        w = .25
        setup.courant_field_data = (np.array([[a, b]]).T, np.array([[0, 0]]))
        setup.positions = [[w], [0]]
        setup.scheme = 'FTFS'
        sut, particles = setup.get_displacement()

        # Act
        sut.calculate_displacement(sut.displacement, sut.courant,
                                   particles.state['cell origin'], particles.state['position in cell'])

        # Assert
        np.testing.assert_equal(sut.displacement[0, 0], (1 - w) * a + w * b)

    def test_calculate_displacement_dim1(self):
        # Arrange
        setup = Setup()
        a = .1
        b = .2
        w = .25
        setup.courant_field_data = (np.array([[0, 0]]).T, np.array([[a, b]]))
        setup.positions = [[0], [w]]
        setup.scheme = 'FTFS'
        sut, particles = setup.get_displacement()

        # Act
        sut.calculate_displacement(sut.displacement, sut.courant,
                                   particles.state['cell origin'], particles.state['position in cell'])

        # Assert
        np.testing.assert_equal(sut.displacement[1, 0], (1 - w) * a + w * b)

    def test_update_position(self):
        # Arrange
        setup = Setup()
        px = .1
        py = .2
        setup.positions = [[px], [py]]
        sut, particles = setup.get_displacement()

        droplet_id = 0
        sut.displacement[0, droplet_id] = .1
        sut.displacement[1, droplet_id] = .2

        # Act
        sut.update_position(particles.state['position in cell'], sut.displacement)

        # Assert
        for d in range(2):
            assert particles.state['position in cell'][d, droplet_id] == (
                    setup.positions[d][droplet_id] + sut.displacement[d, droplet_id]
            )

    def test_update_cell_origin(self):
        # Arrange
        setup = Setup()
        sut, particles = setup.get_displacement()

        droplet_id = 0
        state = particles.state
        state['position in cell'][0, droplet_id] = 1.1
        state['position in cell'][1, droplet_id] = 1.2

        # Act
        sut.update_cell_origin(state['cell origin'], state['position in cell'])

        # Assert
        for d in range(2):
            assert state['cell origin'][d, droplet_id] == setup.positions[d][droplet_id] + 1
            assert state['position in cell'][d, droplet_id] == (state['position in cell'][d, droplet_id]
                                                                - np.floor(state['position in cell'][d, droplet_id]))

    def test_boundary_condition(self):
        # Arrange
        setup = Setup()
        sut, particles = setup.get_displacement()

        droplet_id = 0
        state = particles.state
        state['cell origin'][0, droplet_id] = 1.1
        state['cell origin'][1, droplet_id] = 1.2

        # Act
        sut.boundary_condition(state['cell origin'])

        # Assert
        assert state['cell origin'][0, droplet_id] == 0
        assert state['cell origin'][1, droplet_id] == 0
