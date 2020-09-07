"""
Created at 06.11.2019
"""

from ._moist_eulerian import _MoistEulerian
from PySDM.mesh import Mesh
from .kinematic_2d.arakawa_c import nondivergent_vector_field_2d, make_rhod, courant_field


class MoistEulerian2DKinematic(_MoistEulerian):
    def __init__(self, dt, grid, size, rhod_of, eulerian_advection_solvers):
        super().__init__(dt, Mesh(grid, size), [])

        self.rhod_of = rhod_of
        self.eulerian_advection_solvers = eulerian_advection_solvers

    def register(self, builder):
        super().register(builder)
        rhod = builder.core.Storage.from_ndarray(self.eulerian_advection_solvers.g_factor.ravel())
        self._values["current"]["rhod"] = rhod
        self._tmp["rhod"] = rhod

        super().sync()
        self.notify()

    def _get_thd(self):
        return self.eulerian_advection_solvers['th'].advectee.get()

    def _get_qv(self):
        return self.eulerian_advection_solvers['qv'].advectee.get()

    def step(self):
        self.eulerian_advection_solvers()

    def sync(self):
        self.eulerian_advection_solvers.wait()
        super().sync()

    def get_courant_field_data(self):
        return courant_field(self.eulerian_advection_solvers.advector, self.rhod_of, self.core.mesh.grid)
