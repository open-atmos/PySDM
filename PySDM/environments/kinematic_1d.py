"""
Single-column time-varying-updraft framework with moisture advection handled by
[PyMPDATA](http://github.com/open-atmos/PyMPDATA/)
"""

import numpy as np

from PySDM.environments.impl.moist import Moist

from ..impl import arakawa_c
from ..initialisation.equilibrate_wet_radii import equilibrate_wet_radii


class Kinematic1D(Moist):
    def __init__(self, *, dt, mesh, thd_of_z, rhod_of_z, z0=0):
        super().__init__(dt, mesh, [])
        self.thd0 = thd_of_z(z0 + mesh.dz * arakawa_c.z_scalar_coord(mesh.grid))
        self.rhod = rhod_of_z(z0 + mesh.dz * arakawa_c.z_scalar_coord(mesh.grid))
        self.formulae = None

    def register(self, builder):
        super().register(builder)
        self.formulae = builder.particulator.formulae
        rhod = builder.particulator.Storage.from_ndarray(self.rhod)
        self._values["current"]["rhod"] = rhod
        self._tmp["rhod"] = rhod

    def get_water_vapour_mixing_ratio(self) -> np.ndarray:
        return self.particulator.dynamics["EulerianAdvection"].solvers.advectee.get()

    def get_thd(self) -> np.ndarray:
        return self.thd0

    def init_attributes(
        self,
        *,
        spatial_discretisation,
        spectral_discretisation,
        kappa,
        z_part=None,
        collisions_only=False
    ):
        super().sync()
        self.notify()

        attributes = {}
        with np.errstate(all="raise"):
            positions = spatial_discretisation.sample(
                backend=self.particulator.backend,
                grid=self.mesh.grid,
                n_sd=self.particulator.n_sd,
                z_part=z_part,
            )
            (
                attributes["cell id"],
                attributes["cell origin"],
                attributes["position in cell"],
            ) = self.mesh.cellular_attributes(positions)

            if collisions_only:
                v_wet, n_per_kg = spectral_discretisation.sample(
                    backend=self.particulator.backend, n_sd=self.particulator.n_sd
                )
                # attributes["dry volume"] = v_wet
                attributes["volume"] = v_wet
                # attributes["kappa times dry volume"] = attributes["dry volume"] * kappa
            else:
                r_dry, n_per_kg = spectral_discretisation.sample(
                    backend=self.particulator.backend, n_sd=self.particulator.n_sd
                )
                attributes["dry volume"] = self.formulae.trivia.volume(radius=r_dry)
                attributes["kappa times dry volume"] = attributes["dry volume"] * kappa
                r_wet = equilibrate_wet_radii(
                    r_dry=r_dry,
                    environment=self,
                    cell_id=attributes["cell id"],
                    kappa_times_dry_volume=attributes["kappa times dry volume"],
                )
                attributes["volume"] = self.formulae.trivia.volume(radius=r_wet)

            rhod = self["rhod"].to_ndarray()
            cell_id = attributes["cell id"]
            domain_volume = np.prod(np.array(self.mesh.size))

        attributes["multiplicity"] = n_per_kg * rhod[cell_id] * domain_volume

        return attributes

    @property
    def dv(self):
        return self.mesh.dv
