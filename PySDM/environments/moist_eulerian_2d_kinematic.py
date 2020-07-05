"""
Created at 06.11.2019
"""

import numpy as np
from MPyDATA.factories import Factories
from MPyDATA.arakawa_c.discretisation import z_vec_coord, x_vec_coord
from MPyDATA.options import Options
from ._moist_eulerian import _MoistEulerian
from threading import Thread
from PySDM.mesh import Mesh
from PySDM import ParticlesBuilder


class MoistEulerian2DKinematic(_MoistEulerian):

    def __init__(self, particles_builder: ParticlesBuilder, dt, grid, size, stream_function, field_values, rhod_of,
                 mpdata_iters, mpdata_iga, mpdata_fct, mpdata_tot):
        super().__init__(particles_builder, dt, Mesh(grid, size), [])

        self.__rhod_of = rhod_of

        grid = self.mesh.grid
        rhod = np.repeat(
            rhod_of(
                (np.arange(grid[1]) + 1 / 2) / grid[1]
            ).reshape((1, grid[1])),
            grid[0],
            axis=0
        )

        self.__GC, self.__mpdatas = Factories.stream_function_2d(
            grid=self.mesh.grid, size=self.mesh.size, dt=self.dt,
            stream_function=stream_function,
            field_values=dict((key, np.full(grid, value)) for key, value in field_values.items()),
            g_factor=rhod,
            options=Options(
                n_iters=mpdata_iters,
                infinite_gauge=mpdata_iga,
                flux_corrected_transport=mpdata_fct,
                third_order_terms=mpdata_tot
            )
        )

        rhod = particles_builder.particles.backend.Storage.from_ndarray(rhod.ravel())
        self._values["current"]["rhod"] = rhod
        self._tmp["rhod"] = rhod
        self.asynchronous = False
        self.thread: (Thread, None) = None

        super().sync()
        self.notify()

    def _get_thd(self):
        return self.__mpdatas['th'].curr.get()

    def _get_qv(self):
        return self.__mpdatas['qv'].curr.get()

    def __mpdata_step(self):
        for mpdata in self.__mpdatas.values():
            mpdata.advance(1)

    def step(self):
        if self.asynchronous:
            self.thread = Thread(target=self.__mpdata_step, args=())
            self.thread.start()
        else:
            self.__mpdata_step()

    def wait(self):
        if self.asynchronous:
            if self.thread is not None:
                self.thread.join()

    def sync(self):
        self.wait()
        super().sync()

    def get_courant_field_data(self):
        result = [
            self.__GC.get_component(0) / self.__rhod_of(
                x_vec_coord(self.particles.mesh.grid)[1]),
            self.__GC.get_component(1) / self.__rhod_of(
                z_vec_coord(self.particles.mesh.grid)[1])
        ]
        return result
