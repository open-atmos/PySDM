"""
Single-column time-varying-updraft framework with moisture advection handled by
[PyMPDATA](http://github.com/atmos-cloud-sim-uj/PyMPDATA/)
"""

import numpy as np
from PySDM.environments.impl.moist import Moist
from ..initialisation.equilibrate_wet_radii import equilibrate_wet_radii
from ..impl import arakawa_c


class Kinematic1D(Moist):
    def __init__(self, dt, mesh, thd_of_z, rhod_of_z):
        super().__init__(dt, mesh, [])
        self.thd0 = thd_of_z(mesh.dz * arakawa_c.z_scalar_coord(mesh.grid))
        self.rhod = rhod_of_z(mesh.dz * arakawa_c.z_scalar_coord(mesh.grid))
        self.formulae = None

    def register(self, builder):
        super().register(builder)
        self.formulae = builder.particulator.formulae
        rhod = builder.particulator.Storage.from_ndarray(self.rhod)
        self._values["current"]["rhod"] = rhod
        self._tmp["rhod"] = rhod

    def get_qv(self) -> np.ndarray:
        return self.particulator.dynamics['EulerianAdvection'].solvers.advectee.get()

    def get_thd(self) -> np.ndarray:
        return self.thd0

    def init_attributes(self, *,
                        spatial_discretisation,
                        spectral_discretisation,
                        kappa
                        ):
        super().sync()
        self.notify()

        attributes = {}
        with np.errstate(all='raise'):
            positions = spatial_discretisation.sample(self.mesh.grid, self.particulator.n_sd)
            attributes['cell id'], attributes['cell origin'], attributes['position in cell'] = \
                self.mesh.cellular_attributes(positions)

            r_dry, n_per_kg = spectral_discretisation.sample(self.particulator.n_sd)
            attributes['dry volume'] = self.formulae.trivia.volume(radius=r_dry)
            attributes['kappa times dry volume'] = attributes['dry volume'] * kappa
            r_wet = equilibrate_wet_radii(
                r_dry, self, cell_id=attributes['cell id'],
                kappa_times_dry_volume=attributes['kappa times dry volume']
            )

            rhod = self['rhod'].to_ndarray()
            cell_id = attributes['cell id']
            domain_volume = np.prod(np.array(self.mesh.size))

        attributes['n'] = n_per_kg * rhod[cell_id] * domain_volume
        attributes['volume'] = self.formulae.trivia.volume(radius=r_wet)

        return attributes

    @property
    def dv(self):
        return self.mesh.dv
