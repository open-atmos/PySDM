"""
Created at 23.10.2019
"""

import numpy as np
from PySDM.particles_builder import ParticlesBuilder


class Displacement:

    def __init__(self, particles_builder: ParticlesBuilder, scheme='FTBS', sedimentation=False):
        particles_builder.request_attribute('terminal velocity')
        self.particles = particles_builder.particles
        courant_field = self.particles.environment.get_courant_field_data()
        Storage = self.particles.backend.Storage

        # CFL # TODO: this should be done by MPyDATA
        for d in range(len(courant_field)):
            assert np.amax(abs(courant_field[d])) <= 1

        # TODO: replace with make_calculate_displacement
        if scheme == 'FTFS':
            method = self.particles.backend.explicit_in_space
        elif scheme == 'FTBS':
            method = self.particles.backend.implicit_in_space
        else:
            raise NotImplementedError()

        self.scheme = method
        self.enable_sedimentation = sedimentation
        self.dimension = len(courant_field)
        self.grid = Storage.from_ndarray(
            np.array([courant_field[1].shape[0], courant_field[0].shape[1]], dtype=np.int64))
        self.courant = [Storage.from_ndarray(courant_field[i]) for i in range(self.dimension)]
        self.displacement = Storage.from_ndarray(np.zeros((self.dimension, self.particles.n_sd)))
        self.temp = Storage.from_ndarray(np.zeros((self.dimension, self.particles.n_sd), dtype=np.int64))

    def __call__(self):
        # TIP: not need all array only [idx[:sd_num]]
        displacement = self.displacement
        cell_origin = self.particles.state['cell origin']
        position_in_cell = self.particles.state['position in cell']

        self.calculate_displacement(displacement, self.courant, cell_origin, position_in_cell)
        self.update_position(position_in_cell, displacement)
        if self.enable_sedimentation:
            self.particles.remove_precipitated()
        self.update_cell_origin(cell_origin, position_in_cell)
        self.boundary_condition(cell_origin)
        self.particles.state.recalculate_cell_id()

    def calculate_displacement(self, displacement, courant, cell_origin, position_in_cell):
        for dim in range(self.dimension):
            self.particles.backend.calculate_displacement(
                dim, self.scheme, displacement, courant[dim], cell_origin, position_in_cell)
        if self.enable_sedimentation:
            displacement_z = self.particles.backend.read_row(displacement, self.dimension-1)
            dt_over_dz = self.particles.dt / self.particles.mesh.dz
            self.particles.backend.multiply(displacement_z, 1 / dt_over_dz)
            self.particles.backend.subtract(displacement_z, self.particles.state['terminal velocity'])
            self.particles.backend.multiply(displacement_z, dt_over_dz)

    def update_position(self, position_in_cell, displacement):
        position_in_cell += displacement

    def update_cell_origin(self, cell_origin, position_in_cell):
        floor_of_position = self.temp
        floor_of_position.floor(position_in_cell)
        cell_origin += floor_of_position
        position_in_cell -= floor_of_position

    def boundary_condition(self, cell_origin):
        # TODO: hardcoded periodic
        # TODO: particles above the mesh
        cell_origin %= self.grid
