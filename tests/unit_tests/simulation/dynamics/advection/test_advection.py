"""
Created at 23.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from PySDM.backends.default import Default
from tests.unit_tests.simulation.state.dummy_particles import DummyParticles
from PySDM.simulation.dynamics.advection import Advection
from PySDM.simulation.state.state_factory import StateFactory
import numpy as np
from tests.unit_tests.simulation.dynamics.advection.dummy_environment import DummyEnvironment


class TestExplicitEulerWithInterpolation:

    def test_single_cell(self):

        # Arrange
        n = np.ones(1)
        grid = (1, 1)
        particles = DummyParticles(Default, n_sd=len(n))
        particles.set_mesh(grid)
        particles.set_environment(DummyEnvironment, ((np.array([[.1, .2]]).T, np.array([[.3, .4]])),))
        positions = Default.from_ndarray(np.array([[0.5, 0.5]]))
        cell_id, cell_origin, position_in_cell = particles.mesh.cellular_attributes(positions)
        particles.state = StateFactory.state(n=n, intensive={}, extensive={},
                                             cell_id=cell_id, cell_origin=cell_origin, position_in_cell=position_in_cell,
                                             particles=particles)
        sut = Advection(particles=particles)

        # Act
        sut()

        # Assert
        # TODO

    def test_advection(self):
        n = np.ones(1)
        grid = (3, 3)
        particles = DummyParticles(Default, n_sd=len(n))
        particles.set_mesh(grid)
        particles.set_environment(DummyEnvironment, ((np.ones((4, 3)), np.zeros((3, 4))),))
        positions = Default.from_ndarray(np.array([[1.5, 1.5]]))
        cell_id, cell_origin, position_in_cell = particles.mesh.cellular_attributes(positions)

        particles.state = StateFactory.state(n=n, intensive={}, extensive={},
                                             cell_id=cell_id, cell_origin=cell_origin, position_in_cell=position_in_cell,
                                             particles=particles)
        sut = Advection(particles=particles)

        sut()

        np.testing.assert_array_equal(particles.state.cell_origin[0, :], np.array([2, 1]))

    def test_calculate_displacement(self):
        # Arrange
        n = np.ones(1)
        grid = (1, 1)
        particles = DummyParticles(Default, n_sd=len(n))
        particles.set_mesh(grid)
        a = .1
        b = .2
        w = .25
        particles.set_environment(DummyEnvironment, ((np.array([[a, b]]).T, np.array([[0, 0]])),))
        positions = Default.from_ndarray(np.array([[w, 0]]))
        cell_id, cell_origin, position_in_cell = particles.mesh.cellular_attributes(positions)

        particles.state = StateFactory.state(n=n, intensive={}, extensive={},
                                             cell_id=cell_id, cell_origin=cell_origin,
                                             position_in_cell=position_in_cell,
                                             particles=particles)
        sut = Advection(particles=particles, scheme='FTFS')

        # Act
        sut.calculate_displacement(sut.displacement, sut.courant,
                                   particles.state.cell_origin, particles.state.position_in_cell)

        # Assert
        np.testing.assert_equal(sut.displacement[0, 0], (1 - w) * a + w * b)

    def test_calculate_displacement_dim1(self):
        # Arrange
        n = np.ones(1)
        grid = (1, 1)
        a = .1
        b = .2
        w = .25
        particles = DummyParticles(Default, n_sd=len(n))
        particles.set_mesh(grid)
        particles.set_environment(DummyEnvironment, ((np.array([[0, 0]]).T, np.array([[a, b]])),))
        positions = Default.from_ndarray(np.array([[0, w]]))
        cell_id, cell_origin, position_in_cell = particles.mesh.cellular_attributes(positions)

        particles.state = StateFactory.state(n=n, intensive={}, extensive={},
                                                cell_id=cell_id, cell_origin=cell_origin,
                                                position_in_cell=position_in_cell,
                                                particles=particles)
        sut = Advection(particles=particles, scheme='FTFS')

        # Act
        sut.calculate_displacement(sut.displacement, sut.courant,
                                   particles.state.cell_origin, particles.state.position_in_cell)

        # Assert
        np.testing.assert_equal(sut.displacement[0, 1], (1 - w) * a + w * b)

    def test_update_position(self):
        # Arrange
        n = np.ones(1)
        grid = (1, 1)
        particles = DummyParticles(Default, n_sd=len(n))
        particles.set_mesh(grid)
        particles.set_environment(DummyEnvironment, ((np.array([[0, 0]]).T, np.array([[0, 0]])),))
        droplet_id = 0
        px = .1
        py = .2
        positions = Default.from_ndarray(np.array([[px, py]]))
        cell_id, cell_origin, position_in_cell = particles.mesh.cellular_attributes(positions)

        particles.state = StateFactory.state(n=n, intensive={}, extensive={},
                                             cell_id=cell_id, cell_origin=cell_origin,
                                             position_in_cell=position_in_cell,
                                             particles=particles)
        sut = Advection(particles=particles)
        sut.displacement[droplet_id, 0] = .1
        sut.displacement[droplet_id, 1] = .2

        # Act
        sut.update_position(particles.state.position_in_cell, sut.displacement)

        # Assert
        for d in range(2):
            assert particles.state.position_in_cell[droplet_id, d] == (
                   positions[droplet_id, d] + sut.displacement[droplet_id, d]
            )

    def test_update_cell_origin(self):
        # Arrange
        n = np.ones(1)
        grid = (1, 1)
        particles = DummyParticles(Default, n_sd=len(n))
        particles.set_mesh(grid)
        particles.set_environment(DummyEnvironment, ((np.array([[0, 0]]).T, np.array([[0, 0]])),))
        droplet_id = 0
        positions = Default.from_ndarray(np.array([[0, 0]]))
        cell_id, cell_origin, position_in_cell = particles.mesh.cellular_attributes(positions)

        particles.state = StateFactory.state(n=n, intensive={}, extensive={},
                                             cell_id=cell_id, cell_origin=cell_origin,
                                             position_in_cell=position_in_cell,
                                             particles=particles)
        sut = Advection(particles=particles)
        state = particles.state
        state.position_in_cell[droplet_id, 0] = 1.1
        state.position_in_cell[droplet_id, 1] = 1.2

        # Act
        sut.update_cell_origin(state.cell_origin, state.position_in_cell)

        # Assert
        for d in range(2):
            assert state.cell_origin[droplet_id, d] == positions[droplet_id, d] + 1
            assert state.position_in_cell[droplet_id, d] == (state.position_in_cell[droplet_id, d]
                                                             - np.floor(state.position_in_cell[droplet_id, d]))

    def test_boundary_condition(self):
        # Arrange
        n = np.ones(1)
        grid = (1, 1)
        particles = DummyParticles(Default, n_sd=len(n))
        particles.set_mesh(grid)
        particles.set_environment(DummyEnvironment, ((np.array([[0, 0]]).T, np.array([[0, 0]])),))
        droplet_id = 0
        positions = Default.from_ndarray(np.array([[0, 0]]))
        cell_id, cell_origin, position_in_cell = particles.mesh.cellular_attributes(positions)

        particles.state = StateFactory.state(n=n, intensive={}, extensive={},
                                             cell_id=cell_id, cell_origin=cell_origin, position_in_cell=position_in_cell,
                                             particles=particles)
        sut = Advection(particles=particles)
        state = particles.state
        state.cell_origin[droplet_id, 0] = 1.1
        state.cell_origin[droplet_id, 1] = 1.2

        # Act
        sut.boundary_condition(state.cell_origin)

        # Assert
        assert state.cell_origin[droplet_id, 0] == 0
        assert state.cell_origin[droplet_id, 1] == 0
