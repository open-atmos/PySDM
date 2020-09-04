"""
Created at 06.11.2019
"""

import numpy as np
from MPyDATA.arakawa_c.discretisation import z_vec_coord, x_vec_coord
from ._moist_eulerian import _MoistEulerian
from threading import Thread
from PySDM.mesh import Mesh
from .kinematic_2d.arakawa_c import nondivergent_vector_field_2d, make_rhod

from .kinematic_2d.mpdata import make_advection_solver


class MoistEulerian2DKinematic(_MoistEulerian):
    def __init__(self, dt, grid, size, stream_function, field_values, rhod_of,
                 mpdata_iters, mpdata_iga, mpdata_fct, mpdata_tot):
        super().__init__(dt, Mesh(grid, size), [])

        self.__rhod_of = rhod_of

        grid = self.mesh.grid
        self.rhod = make_rhod(grid, rhod_of)
        rho_times_courant = nondivergent_vector_field_2d(grid, size, dt, stream_function)
        self.__advector, self.__mpdatas = make_advection_solver(
            grid=self.mesh.grid, dt=self.dt,
            field_values=dict((key, np.full(grid, value)) for key, value in field_values.items()),
            g_factor=self.rhod,
            rho_times_courant=rho_times_courant,
            mpdata_iters=mpdata_iters,
            mpdata_infinite_gauge=mpdata_iga,
            mpdata_flux_corrected_transport=mpdata_fct,
            mpdata_third_order_terms=mpdata_tot
        )

        self.asynchronous = False
        self.thread: (Thread, None) = None

    def register(self, builder):
        super().register(builder)
        rhod = builder.core.Storage.from_ndarray(self.rhod.ravel())
        self._values["current"]["rhod"] = rhod
        self._tmp["rhod"] = rhod
        delattr(self, 'rhod')

        super().sync()
        self.notify()

    def _get_thd(self):
        return self.__mpdatas['th'].advectee.get()

    def _get_qv(self):
        return self.__mpdatas['qv'].advectee.get()

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
        rho_times_courant = (
            self.__advector.get_component(0),
            self.__advector.get_component(1)
        )
        Z_COORD=1
        result = (
            rho_times_courant[0] / self.__rhod_of(zZ=x_vec_coord(self.core.mesh.grid)[Z_COORD]),
            rho_times_courant[1] / self.__rhod_of(zZ=z_vec_coord(self.core.mesh.grid)[Z_COORD])
        )
        return result
