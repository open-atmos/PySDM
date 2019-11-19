"""
Created at 23.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from PySDM.backends.default import Default
from tests.unit_tests.simulation.state.dummy_simulation import DummySimulation
from PySDM.simulation.dynamics.advection import Advection
from PySDM.simulation.state.state_factory import StateFactory
import numpy as np


class TestExplicitEulerWithInterpolation:

    def test_single_cell(self):

        # Arrange
        n = np.ones(1)
        simulation = DummySimulation(Default)
        simulation.add_attrs(n_sd=len(n))
        positions = Default.from_ndarray(np.array([[0.5, 0.5]]))
        courant_field = (np.array([[.1, .2]]).T, np.array([[.3, .4]]))
        simulation.state = StateFactory.state_2d(n=n, grid=(1, 1), intensive={}, extensive={},
                                                 positions=positions, simulation=simulation)
        sut = Advection(courant_field=courant_field, simulation=simulation)

        # Act
        sut()

        # Assert
        # TODO

    def test_advection(self):
        n = np.ones(1)
        simulation = DummySimulation(Default)
        simulation.add_attrs(n_sd=len(n))
        positions = Default.from_ndarray(np.array([[1.5, 1.5]]))
        courant_field = (np.ones((4, 3)), np.zeros((3, 4)))
        simulation.state = StateFactory.state_2d(n=n, grid=(3, 3), intensive={}, extensive={}, 
                                                 simulation=simulation, positions=positions)
        sut = Advection(courant_field=courant_field, simulation=simulation)

        sut()

        np.testing.assert_array_equal(simulation.state.cell_origin[0, :], np.array([2, 1]))

    def test_calculate_displacement(self):
        # Arrange
        n = np.ones(1)
        simulation = DummySimulation(Default)
        simulation.add_attrs(n_sd=len(n))
        a = .1
        b = .2
        w = .25
        positions = Default.from_ndarray(np.array([[w, 0]]))
        courant_field = (np.array([[a, b]]).T, np.array([[0, 0]]))
        simulation.state = StateFactory.state_2d(n=n, grid=(1, 1), intensive={}, extensive={}, 
                                                 simulation=simulation, positions=positions)
        sut = Advection(courant_field=courant_field, simulation=simulation, scheme='FTFS')

        # Act
        sut.calculate_displacement(sut.displacement, sut.courant, simulation.state.cell_origin, simulation.state.position_in_cell)

        # Assert
        np.testing.assert_equal(sut.displacement[0, 0], (1 - w) * a + w * b)

    def test_calculate_displacement_dim1(self):
        # Arrange
        n = np.ones(1)
        simulation = DummySimulation(Default)
        simulation.add_attrs(n_sd=len(n))
        a = .1
        b = .2
        w = .25
        positions = Default.from_ndarray(np.array([[0, w]]))
        courant_field = (np.array([[0, 0]]).T, np.array([[a, b]]))
        simulation.state = StateFactory.state_2d(n=n, grid=(1, 1), intensive={}, extensive={}, 
                                                 simulation=simulation, positions=positions)
        sut = Advection(courant_field=courant_field, simulation=simulation, scheme='FTFS')

        # Act
        sut.calculate_displacement(sut.displacement, sut.courant, simulation.state.cell_origin, simulation.state.position_in_cell)

        # Assert
        np.testing.assert_equal(sut.displacement[0, 1], (1 - w) * a + w * b)


    def test_update_position(self):
        # Arrange
        n = np.ones(1)
        simulation = DummySimulation(Default)
        simulation.add_attrs(n_sd=len(n))
        droplet_id = 0
        px = .1
        py = .2
        initial_position = Default.from_ndarray(np.array([[px, py]]))
        dummy_courant_field = (np.array([[0, 0]]).T, np.array([[0, 0]]))
        # TODO: state not needed?
        simulation.state = StateFactory.state_2d(n=n, grid=(1, 1), intensive={}, extensive={}, 
                                                 simulation=simulation, positions=initial_position)
        sut = Advection(courant_field=dummy_courant_field, simulation=simulation)
        sut.displacement[droplet_id, 0] = .1
        sut.displacement[droplet_id, 1] = .2

        # Act
        sut.update_position(simulation.state.position_in_cell, sut.displacement)

        # Assert
        for d in range(2):
            assert simulation.state.position_in_cell[droplet_id, d] == (
                    initial_position[droplet_id, d] + sut.displacement[droplet_id, d]
            )

    def test_update_cell_origin(self):
        # Arrange
        n = np.ones(1)
        simulation = DummySimulation(Default)
        simulation.add_attrs(n_sd=len(n))
        droplet_id = 0
        initial_position = Default.from_ndarray(np.array([[0, 0]]))
        dummy_courant_field = (np.array([[0, 0]]).T, np.array([[0, 0]]))
        # TODO: state not needed?
        simulation.state = StateFactory.state_2d(n=n, grid=(1, 1), intensive={}, extensive={}, 
                                                 simulation=simulation, positions=initial_position)
        sut = Advection(courant_field=dummy_courant_field, simulation=simulation)
        state = simulation.state
        state.position_in_cell[droplet_id, 0] = 1.1
        state.position_in_cell[droplet_id, 1] = 1.2

        # Act
        sut.update_cell_origin(state.cell_origin, state.position_in_cell)

        # Assert
        for d in range(2):
            assert state.cell_origin[droplet_id, d] == initial_position[droplet_id, d] + 1
            assert state.position_in_cell[droplet_id, d] == state.position_in_cell[droplet_id, d] - np.floor(state.position_in_cell[droplet_id, d])

    def test_boundary_condition(self):
        # Arrange
        n = np.ones(1)
        simulation = DummySimulation(Default)
        simulation.add_attrs(n_sd=len(n))
        droplet_id = 0
        initial_position = Default.from_ndarray(np.array([[0, 0]]))
        dummy_courant_field = (np.array([[0, 0]]).T, np.array([[0, 0]]))
        # TODO: state not needed?
        simulation.state = StateFactory.state_2d(n=n, grid=(1, 1), intensive={}, extensive={}, 
                                                 simulation=simulation, positions=initial_position)
        sut = Advection(courant_field=dummy_courant_field, simulation=simulation)
        state = simulation.state
        state.cell_origin[droplet_id, 0] = 1.1
        state.cell_origin[droplet_id, 1] = 1.2

        # Act
        sut.boundary_condition(state.cell_origin)

        # Assert
        assert state.cell_origin[droplet_id, 0] == 0
        assert state.cell_origin[droplet_id, 1] == 0
