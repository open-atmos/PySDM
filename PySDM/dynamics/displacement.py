"""
Particle displacement due to advection by the flow & sedimentation
"""
import numpy as np


class Displacement:
    def __init__(self, enable_sedimentation=False):
        self.core = None
        self.enable_sedimentation = enable_sedimentation
        self.dimension = None
        self.grid = None
        self.courant = None
        self.displacement = None
        self.temp = None
        self.precipitation_in_last_step = 0

    def register(self, builder):
        builder.request_attribute('terminal velocity')
        self.core = builder.core
        self.dimension = len(builder.core.environment.mesh.grid)
        self.grid = self.core.Storage.from_ndarray(np.array(builder.core.environment.mesh.grid, dtype=np.int64))
        if self.dimension == 1:
            courant_field = (np.full(self.grid[0]+1, np.nan),)
        elif self.dimension == 2:
            courant_field = (
                np.full((self.grid[0]+1, self.grid[1]), np.nan),
                np.full((self.grid[0], self.grid[1]+1), np.nan),
            )
        else:
            raise NotImplementedError()
        self.courant = tuple(self.core.Storage.from_ndarray(courant_field[i]) for i in range(self.dimension))
        self.displacement = self.core.Storage.from_ndarray(np.zeros((self.dimension, self.core.n_sd)))
        self.temp = self.core.Storage.from_ndarray(np.zeros((self.dimension, self.core.n_sd), dtype=np.int64))

    def upload_courant_field(self, courant_field):
        for i, component in enumerate(courant_field):
            self.courant[i].upload(component)

    def __call__(self):
        # TIP: not need all array only [idx[:sd_num]]
        cell_origin = self.core.particles['cell origin']
        position_in_cell = self.core.particles['position in cell']

        self.calculate_displacement(self.displacement, self.courant, cell_origin, position_in_cell)
        self.update_position(position_in_cell, self.displacement)
        if self.enable_sedimentation:
            self.precipitation_in_last_step = self.core.particles.remove_precipitated()
        self.update_cell_origin(cell_origin, position_in_cell)
        self.boundary_condition(cell_origin)
        self.core.particles.recalculate_cell_id()

        # TODO #443
        self.core.particles.attributes['position in cell'].mark_updated()
        self.core.particles.attributes['cell origin'].mark_updated()
        self.core.particles.attributes['cell id'].mark_updated()

    def calculate_displacement(self, displacement, courant, cell_origin, position_in_cell):
        for dim in range(self.dimension):
            self.core.bck.calculate_displacement(
                dim, displacement, courant[dim], cell_origin, position_in_cell)
        if self.enable_sedimentation:
            displacement_z = displacement[self.dimension - 1, :]
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
        # TODO #414 hardcoded periodic
        # TODO #414 droplets above the mesh
        cell_origin %= self.grid
