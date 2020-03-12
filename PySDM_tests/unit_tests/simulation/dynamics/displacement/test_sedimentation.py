"""
Created at 12.03.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.backends.default import Default
from PySDM_tests.unit_tests.simulation.state.dummy_particles import DummyParticles
from PySDM.simulation.dynamics.displacement import Displacement
from PySDM.simulation.state.state_factory import StateFactory
import numpy as np
from PySDM_tests.unit_tests.simulation.dynamics.displacement.dummy_environment import DummyEnvironment


class ConstantTerminalVelocity:
    def __init__(self, particles):
        self.values = np.full(particles.n_sd, 1000)


class TestSedimentation:
    def test_boundary_condition(self):
        # Arrange
        n = np.ones(1, dtype=np.int64)
        grid = (1, 1)
        particles = DummyParticles(Default, n_sd=len(n), dt=1)
        particles.set_mesh(grid)
        particles.set_environment(DummyEnvironment, ((np.array([[0, 0]]).T, np.array([[0, 0]])),))
        positions = Default.from_ndarray(np.array([[0, 0]]))
        cell_id, cell_origin, position_in_cell = particles.mesh.cellular_attributes(positions)

        particles.state = StateFactory.state(n=n, intensive={}, extensive={},
                                             cell_id=cell_id, cell_origin=cell_origin, position_in_cell=position_in_cell,
                                             particles=particles)
        sut = Displacement(particles=particles, sedimentation=True)
        particles.set_terminal_velocity(ConstantTerminalVelocity)


        # Act
        sut()

        # Assert
        assert particles.state.SD_num == 0
