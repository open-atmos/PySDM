"""
Created at 23.10.2019
"""

import numpy as np


class Displacement:

    def __init__(self, scheme='FTBS', sedimentation=False):
        self.core = None
        self.scheme = scheme
        self.enable_sedimentation = sedimentation
        self.dimension = None
        self.grid = None
        self.courant = None
        self.displacement = None
        self.temp = None

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

        courant_field = self.core.environment.get_courant_field_data()
        # CFL # TODO: this should be done by MPyDATA
        for d in range(len(courant_field)):
            assert np.amax(abs(courant_field[d])) <= 1

        self.dimension = len(courant_field)
        self.grid = self.core.Storage.from_ndarray(
            np.array([courant_field[1].shape[0], courant_field[0].shape[1]], dtype=np.int64))
        self.courant = [self.core.Storage.from_ndarray(courant_field[i]) for i in range(self.dimension)]
        self.displacement = self.core.Storage.from_ndarray(np.zeros((self.dimension, self.core.n_sd)))
        self.temp = self.core.Storage.from_ndarray(np.zeros((self.dimension, self.core.n_sd), dtype=np.int64))

    def __call__(self):
        # TIP: not need all array only [idx[:sd_num]]
        displacement = self.displacement
        cell_origin = self.core.state['cell origin']
        position_in_cell = self.core.state['position in cell']

        self.calculate_displacement(displacement, self.courant, cell_origin, position_in_cell)
        self.update_position(position_in_cell, displacement)
        if self.enable_sedimentation:
            self.core.state.remove_precipitated()
        self.update_cell_origin(cell_origin, position_in_cell)
        self.boundary_condition(cell_origin)
        self.core.state.recalculate_cell_id()

    def calculate_displacement(self, displacement, courant, cell_origin, position_in_cell):
        for dim in range(self.dimension):
            self.core.bck.calculate_displacement(
                dim, self.scheme, displacement, courant[dim], cell_origin, position_in_cell)
        if self.enable_sedimentation:
            displacement_z = displacement.read_row(self.dimension - 1)
            dt_over_dz = self.core.dt / self.core.mesh.dz
            displacement_z *= 1 / dt_over_dz
            displacement_z -= self.core.state['terminal velocity']
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
