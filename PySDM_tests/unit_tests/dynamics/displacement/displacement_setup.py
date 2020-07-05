"""
Created at 29.04.2020
"""

from PySDM.backends.default import Default
from PySDM_tests.unit_tests.state.dummy_particles import DummyParticles
from PySDM.dynamics import Displacement
from PySDM.state.state_factory import StateFactory
import numpy as np
from PySDM_tests.unit_tests.state.dummy_environment import DummyEnvironment


class Setup:
    def __init__(self):
        self.n = np.ones(1, dtype=np.int64)
        self.grid = (1, 1)
        self.courant_field_data = (np.array([[0, 0]]).T, np.array([[0, 0]]))
        self.positions = [[0], [0]]
        self.scheme = 'FTBS'
        self.sedimentation = False
        self.dt = None

    def get_displacement(self):
        particles = DummyParticles(Default, n_sd=len(self.n))
        particles.set_environment(DummyEnvironment,
                                  {'dt': self.dt,
                                   'grid': self.grid,
                                   'courant_field_data': self.courant_field_data})
        positions = np.array(self.positions)
        cell_id, cell_origin, position_in_cell = particles.mesh.cellular_attributes(positions)
        attributes = {'n': self.n, 'cell id': cell_id, 'cell origin': cell_origin, 'position in cell': position_in_cell}
        particles.get_particles(attributes)
        sut = Displacement(particles_builder=particles, scheme=self.scheme, sedimentation=self.sedimentation)

        return sut, particles
