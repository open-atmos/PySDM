"""
Created at 06.11.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import numpy as np
from MPyDATA.mpdata_factory import MPDATAFactory
from MPyDATA.arakawa_c.discretisation import z_vec_coord, x_vec_coord
from MPyDATA.options import Options
from ._moist_eulerian import _MoistEulerian
from threading import Thread
from .products.relative_humidity import RelativeHumidity
from .products.dry_air_potential_temperature import DryAirPotentialTemperature
from .products.water_vapour_mixing_ratio import WaterVapourMixingRatio
from .products.dry_air_density import DryAirDensity


class MoistEulerian2DKinematic(_MoistEulerian):

    def __init__(self, particles, stream_function, field_values, rhod_of,
                 mpdata_iters, mpdata_iga, mpdata_fct, mpdata_tot):
        super().__init__(particles, [])

        self.__rhod_of = rhod_of

        grid = particles.mesh.grid
        rhod = np.repeat(
            rhod_of(
                (np.arange(grid[1]) + 1 / 2) / grid[1]
            ).reshape((1, grid[1])),
            grid[0],
            axis=0
        )

        self.__GC, self.__mpdatas = MPDATAFactory.stream_function_2d(
            grid=self.particles.mesh.grid, size=self.particles.mesh.size, dt=particles.dt,
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

        rhod = particles.backend.from_ndarray(rhod.ravel())
        self._values["current"]["rhod"] = rhod
        self._tmp["rhod"] = rhod
        self.products = [DryAirDensity(self), RelativeHumidity(self), DryAirPotentialTemperature(self), WaterVapourMixingRatio(self)]
        self.thread: Thread = None

        super().sync()
        self.post_step()

    def _get_thd(self):
        return self.__mpdatas['th'].curr.get()

    def _get_qv(self):
        return self.__mpdatas['qv'].curr.get()

    # @property
    # def eulerian_fields(self):
    #     return self.__eulerian_fields

    def __mpdata_step(self):
        for mpdata in self.__mpdatas.values():
            mpdata.step(1)

    def step(self):
        # self.thread = Thread(target=self.__mpdata_step, args=())
        # self.thread.start()
        self.__mpdata_step()
        pass

    def wait(self):
        # if self.thread is not None:
        #     self.thread.join()
        pass

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
