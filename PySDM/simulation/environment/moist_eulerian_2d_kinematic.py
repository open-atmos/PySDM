"""
Created at 06.11.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import numpy as np
from MPyDATA.mpdata_factory import MPDATAFactory, z_vec_coord, x_vec_coord
from MPyDATA.options import Options
from ._moist_eulerian import _MoistEulerian


class MoistEulerian2DKinematic(_MoistEulerian):

    def __init__(self, particles, stream_function, field_values, rhod_of,
                 mpdata_iters, mpdata_iga, mpdata_fct, mpdata_tot):
        super().__init__(particles, [])

        self.__rhod_of = rhod_of
        self.mpdata_iters = mpdata_iters

        grid = particles.mesh.grid
        rhod = np.repeat(
            rhod_of(
                (np.arange(grid[1]) + 1 / 2) / grid[1]
            ).reshape((1, grid[1])),
            grid[0],
            axis=0
        )

        self.__GC, self.__eulerian_fields = MPDATAFactory.kinematic_2d(
            grid=self.particles.mesh.grid, size=self.particles.mesh.size, dt=particles.dt,
            stream_function=stream_function,
            field_values=field_values,
            g_factor=rhod,
            opts=Options(nug=True, iga=mpdata_iga, fct=mpdata_fct, tot=mpdata_tot)
        )

        rhod = particles.backend.from_ndarray(rhod.ravel())
        self._values["current"]["rhod"] = rhod
        self._tmp["rhod"] = rhod

        self.sync()
        self.post_step()

    def _get_thd(self):
        return self.__eulerian_fields.mpdatas['th'].arrays.curr.get()

    def _get_qv(self):
        return self.__eulerian_fields.mpdatas['qv'].arrays.curr.get()

    @property
    def eulerian_fields(self):
        return self.__eulerian_fields

    def __mpdata_step(self):
        self.__eulerian_fields.step(n_iters=self.mpdata_iters)

    def step(self):
        self.__mpdata_step()

    def wait(self):
        # TODO
        pass

    def sync(self):
        self.wait()
        super().sync()

    def get_courant_field_data(self):
        result = [
            self.__GC.get_component(0) / self.__rhod_of(
                x_vec_coord(self.particles.mesh.grid, self.particles.mesh.size)[1]),
            self.__GC.get_component(1) / self.__rhod_of(
                z_vec_coord(self.particles.mesh.grid, self.particles.mesh.size)[1])
        ]
        return result
