"""
Created at 06.11.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import numpy as np
from MPyDATA.mpdata.mpdata_factory import MPDATAFactory, z_vec_coord, x_vec_coord
from PySDM.simulation.environment.enviroment import Environment


class Kinematic2D(Environment):
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

        self.thd_lambda = lambda: self.eulerian_fields.mpdatas["th"].curr.get()
        self.qv_lambda = lambda: self.eulerian_fields.mpdatas["qv"].curr.get()

        self.rhod = particles.backend.from_ndarray(rhod.ravel())

        # TODO
        self.ante_step = self.eulerian_fields.step
        self.sync()
        self.post_step()

    # TODO: this is only used from within PySDM, examples always use ["old"] - awkward
    def __getitem__(self, index):
        values = self._values[index]
        if values is None:
            raise Exception("condensation not called.")
        return values

    def post_step(self):
        self.particles.backend.download(self._values["new"]["qv"].reshape(self.particles.mesh.grid), self.qv_lambda())
        self.particles.backend.download(self._values["new"]["thd"].reshape(self.particles.mesh.grid), self.thd_lambda())
        self._swap()
        
    def get_courant_field_data(self):
        result = [  # TODO: test it!!!!
            self.GC.data(0) / self.rhod_of(
                x_vec_coord(self.particles.mesh.grid, self.particles.mesh.size)[1]),
            self.GC.data(1) / self.rhod_of(
                z_vec_coord(self.particles.mesh.grid, self.particles.mesh.size)[1])
        ]
        return result
