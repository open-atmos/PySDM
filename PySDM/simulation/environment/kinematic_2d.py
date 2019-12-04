"""
Created at 06.11.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import numpy as np
from MPyDATA.mpdata_factory import MPDATAFactory, z_vec_coord, x_vec_coord
from PySDM.simulation.environment._moist_air_environment import _MoistAirEnvironment


class Kinematic2D(_MoistAirEnvironment):

    def __init__(self, particles, stream_function, field_values, rhod_of):
        super().__init__(particles, [])

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

        rhod = particles.backend.from_ndarray(rhod.ravel())
        self._values["current"]["rhod"] = rhod
        self._tmp["rhod"] = rhod

        self.sync()
        self.post_step()

    def _get_thd(self):
        return self.eulerian_fields.mpdatas['th'].curr.get()

    def _get_qv(self):
        return self.eulerian_fields.mpdatas['qv'].curr.get()

    # TODO: move back to mesh
    @property
    def dv(self):
        return self.particles.mesh.dv

    def wait(self):
        # TODO
        pass

    def sync(self):
        self.wait()
        super().sync()

    def get_courant_field_data(self):
        result = [  # TODO: test it!!!!
            self.GC.data(0) / self.rhod_of(
                x_vec_coord(self.particles.mesh.grid, self.particles.mesh.size)[1]),
            self.GC.data(1) / self.rhod_of(
                z_vec_coord(self.particles.mesh.grid, self.particles.mesh.size)[1])
        ]
        return result
