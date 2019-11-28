"""
Created at 06.11.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import numpy as np
from MPyDATA.mpdata.mpdata_factory import MPDATAFactory, z_vec_coord, x_vec_coord
from PySDM.simulation.environment.moist_air import MoistAir


class Kinematic2D(MoistAir):
    def __init__(self, particles, stream_function, field_values, rhod_of):
        super().__init__(particles, ['qv', 'thd', 'RH', 'p', 'T'])

        self.rhod_of = rhod_of

        grid = self.particles.mesh.grid
        rhod = np.repeat(
            rhod_of(
                (np.arange(grid[1]) + 1 / 2) / grid[1]
            ).reshape((1, grid[1])),
            grid[0],
            axis=0
        )

        self.GC, self.eulerian_fields = MPDATAFactory.kinematic_2d(
            grid=self.particles.mesh.grid, size=self.particles.mesh.size, dt=particles.dt,
            stream_function=stream_function,
            field_values=field_values,
            g_factor=rhod
        )

        self.thd_lambda = lambda: self.eulerian_fields.mpdatas['th'].curr.get()
        self.qv_lambda = lambda: self.eulerian_fields.mpdatas['qv'].curr.get()

        self.rhod = particles.backend.from_ndarray(rhod.ravel())

        self.sync()
        self.post_step()

    def ante_step(self):
        self.eulerian_fields.step()

    def post_step(self):
        self.particles.backend.download(self.get_predicted('qv').reshape(self.particles.mesh.grid), self.qv_lambda())
        self.particles.backend.download(self.get_predicted('thd').reshape(self.particles.mesh.grid), self.thd_lambda())
        self._update()
        
    def get_courant_field_data(self):
        result = [  # TODO: test it!!!!
            self.GC.data(0) / self.rhod_of(
                x_vec_coord(self.particles.mesh.grid, self.particles.mesh.size)[1]),
            self.GC.data(1) / self.rhod_of(
                z_vec_coord(self.particles.mesh.grid, self.particles.mesh.size)[1])
        ]
        return result
