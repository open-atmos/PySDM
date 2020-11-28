"""
Created at 23.10.2019
"""

import numpy as np


class Displacement:
    # TODO: create a separate Sedimentation dynamic that links in to Displacement
    def __init__(self, courant_field, scheme='FTBS', enable_sedimentation=False):
        self.core = None
        self.scheme = scheme
        self.enable_sedimentation = enable_sedimentation
        self.dimension = None
        self.grid = None
        self.courant = None
        self.displacement = None
        self.temp = None
        self.courant_field = courant_field
        self.precipitation_in_last_step = 0

    def register(self, builder):
        builder.request_attribute('terminal velocity')
        self.core = builder.core

        # TODO: replace with make_calculate_displacement
        if self.scheme == 'FTFS':
            method = self.core.backend.explicit_in_space
        elif self.scheme == 'FTBS':
            method = self.core.backend.implicit_in_space
        else:
            raise NotImplementedError()
        self.scheme = method

        self.dimension = len(self.courant_field)
        # TODO: simplification
        self.grid = self.core.Storage.from_ndarray(
            np.array([self.courant_field[1].shape[0], self.courant_field[0].shape[1]], dtype=np.int64))
        self.courant = [self.core.Storage.from_ndarray(self.courant_field[i]) for i in range(self.dimension)]
        self.displacement = self.core.Storage.from_ndarray(np.zeros((self.dimension, self.core.n_sd)))
        self.temp = self.core.Storage.from_ndarray(np.zeros((self.dimension, self.core.n_sd), dtype=np.int64))

    def __call__(self):
        # TIP: not need all array only [idx[:sd_num]]
        displacement = self.displacement
        cell_origin = self.core.particles['cell origin']
        position_in_cell = self.core.particles['position in cell']

        self.calculate_displacement(displacement, self.courant, cell_origin, position_in_cell)
        self.update_position(position_in_cell, displacement)
        if self.enable_sedimentation:
            self.precipitation_in_last_step = self.core.particles.remove_precipitated()
        self.update_cell_origin(cell_origin, position_in_cell)
        self.boundary_condition(cell_origin)
        self.core.particles.recalculate_cell_id()

    def calculate_displacement(self, displacement, courant, cell_origin, position_in_cell):
        for dim in range(self.dimension):
            self.core.bck.calculate_displacement(
                dim, self.scheme, displacement, courant[dim], cell_origin, position_in_cell)
        if self.enable_sedimentation:
            displacement_z = displacement.read_row(self.dimension - 1)
            dt_over_dz = self.core.dt / self.core.mesh.dz
            displacement_z *= 1 / dt_over_dz
            displacement_z -= self.core.particles['terminal velocity']
            displacement_z *= dt_over_dz

    def update_position(self, position_in_cell, displacement):
        position_in_cell += displacement

    def update_cell_origin(self, cell_origin, position_in_cell):
        floor_of_position = self.temp
        floor_of_position.floor(position_in_cell)
        cell_origin += floor_of_position
        position_in_cell -= floor_of_position

    def boundary_condition(self, cell_origin):
        # TODO: hardcoded periodic
        # TODO: droplets above the mesh
        cell_origin %= self.grid
