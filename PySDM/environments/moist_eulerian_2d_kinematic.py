"""
Created at 06.11.2019
"""

from ._moist_eulerian import _MoistEulerian
from PySDM.mesh import Mesh
from .kinematic_2d.arakawa_c import courant_field
import numpy as np
from PySDM.initialisation.r_wet_init import r_wet_init_impl
from PySDM.initialisation.temperature_init import temperature_init
from PySDM.initialisation.multiplicities import discretise_n
from PySDM.physics import formulae as phys


class MoistEulerian2DKinematic(_MoistEulerian):
    def __init__(self, dt, grid, size, rhod_of):
        super().__init__(dt, Mesh(grid, size), [])
        self.rhod_of = rhod_of
        self.eulerian_advection_solvers = None

    def set_advection_solver(self, eulerian_advection_solvers):
        self.eulerian_advection_solvers = eulerian_advection_solvers

    def register(self, builder):
        assert self.eulerian_advection_solvers is not None
        super().register(builder)
        rhod = builder.core.Storage.from_ndarray(self.eulerian_advection_solvers.g_factor.ravel())
        self._values["current"]["rhod"] = rhod
        self._tmp["rhod"] = rhod

        super().sync()
        self.notify()

    def init_attributes(self, *,
        spatial_discretisation,
        spectral_discretisation,
        kappa,
        enable_temperatures=False
        ):
        attributes = {}
        with np.errstate(all='raise'):
            positions = spatial_discretisation(self.mesh.grid, self.core.n_sd)
            attributes['cell id'], attributes['cell origin'], attributes['position in cell'] = \
                self.mesh.cellular_attributes(positions)
            r_dry, n_per_kg = spectral_discretisation(self.core.n_sd)
            T = self['T'].to_ndarray()
            p = self['p'].to_ndarray()
            RH = self['RH'].to_ndarray()
            r_wet = r_wet_init_impl(r_dry, T, p, RH, attributes['cell id'], kappa)
            rhod = self['rhod'].to_ndarray()
            cell_id = attributes['cell id']
            domain_volume = np.prod(np.array(self.mesh.size))

        if enable_temperatures:
            attributes['temperature'] = temperature_init(self, attributes['cell id'])
        attributes['n'] = discretise_n(n_per_kg * rhod[cell_id] * domain_volume)
        attributes['volume'] = phys.volume(radius=r_wet)
        attributes['dry volume'] = phys.volume(radius=r_dry)

        return attributes


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
