"""
Created at 06.11.2019
"""

from ._moist import _Moist
from PySDM.state.mesh import Mesh
from ..state import arakawa_c
import numpy as np
from PySDM.initialisation.r_wet_init import r_wet_init, default_rtol
from PySDM.initialisation.temperature_init import temperature_init
from PySDM.initialisation.multiplicities import discretise_n
from PySDM.physics import formulae as phys


class Kinematic2D(_Moist):
    def __init__(self, dt, grid, size, rhod_of, field_values):
        super().__init__(dt, Mesh(grid, size), [])
        self.rhod_of = rhod_of
        self.field_values = field_values

    def register(self, builder):
        super().register(builder)
        rhod = builder.core.Storage.from_ndarray(arakawa_c.make_rhod(self.mesh.grid, self.rhod_of).ravel())
        self._values["current"]["rhod"] = rhod
        self._tmp["rhod"] = rhod

    @property
    def dv(self):
        return self.mesh.dv

    def init_attributes(self, *,
                        spatial_discretisation,
                        spectral_discretisation,
                        kappa,
                        enable_temperatures=False,
                        rtol=default_rtol
                        ):
        # TODO move to one method
        super().sync()
        self.notify()

        attributes = {}
        with np.errstate(all='raise'):
            positions = spatial_discretisation.sample(self.mesh.grid, self.core.n_sd)
            attributes['cell id'], attributes['cell origin'], attributes['position in cell'] = \
                self.mesh.cellular_attributes(positions)
            r_dry, n_per_kg = spectral_discretisation.sample(self.core.n_sd)
            r_wet = r_wet_init(r_dry, self, attributes['cell id'], kappa, rtol)
            rhod = self['rhod'].to_ndarray()
            cell_id = attributes['cell id']
            domain_volume = np.prod(np.array(self.mesh.size))

        if enable_temperatures:
            attributes['temperature'] = temperature_init(self, attributes['cell id'])
        attributes['n'] = discretise_n(n_per_kg * rhod[cell_id] * domain_volume)
        attributes['volume'] = phys.volume(radius=r_wet)
        attributes['dry volume'] = phys.volume(radius=r_dry)

        return attributes

    def get_thd(self):
        return self.core.dynamics['EulerianAdvection'].solvers['th'].advectee.get()

    def get_qv(self):
        return self.core.dynamics['EulerianAdvection'].solvers['qv'].advectee.get()

    def sync(self):
        self.core.dynamics['EulerianAdvection'].solvers.wait()
        super().sync()
