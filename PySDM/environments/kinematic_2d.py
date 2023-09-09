"""
Two-dimensional single-eddy prescribed-flow framework with moisture and heat advection
handled by [PyMPDATA](http://github.com/open-atmos/PyMPDATA/)
"""

import numpy as np

from PySDM.environments.impl.moist import Moist
from PySDM.impl.mesh import Mesh
from PySDM.initialisation.equilibrate_wet_radii import (
    default_rtol,
    equilibrate_wet_radii,
)
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity

from ..impl import arakawa_c


class Kinematic2D(Moist):
    def __init__(self, *, dt, grid, size, rhod_of, mixed_phase=False):
        super().__init__(dt, Mesh(grid, size), [], mixed_phase=mixed_phase)
        self.rhod_of = rhod_of
        self.formulae = None

    def register(self, builder):
        super().register(builder)
        self.formulae = builder.particulator.formulae
        rhod = builder.particulator.Storage.from_ndarray(
            arakawa_c.make_rhod(self.mesh.grid, self.rhod_of).ravel()
        )
        self._values["current"]["rhod"] = rhod
        self._tmp["rhod"] = rhod

    @property
    def dv(self):
        return self.mesh.dv

    def init_attributes(
        self,
        *,
        spatial_discretisation,
        kappa,
        dry_radius_spectrum,
        rtol=default_rtol,
        n_sd=None,
        spectral_sampling=ConstantMultiplicity
    ):
        super().sync()
        self.notify()
        n_sd = n_sd or self.particulator.n_sd
        attributes = {}
        with np.errstate(all="raise"):
            positions = spatial_discretisation.sample(
                backend=self.particulator.backend, grid=self.mesh.grid, n_sd=n_sd
            )
            (
                attributes["cell id"],
                attributes["cell origin"],
                attributes["position in cell"],
            ) = self.mesh.cellular_attributes(positions)

            r_dry, n_per_kg = spectral_sampling(spectrum=dry_radius_spectrum).sample(
                n_sd=n_sd, backend=self.particulator.backend
            )

            attributes["dry volume"] = self.formulae.trivia.volume(radius=r_dry)
            attributes["kappa times dry volume"] = kappa * attributes["dry volume"]
            if kappa == 0:
                r_wet = r_dry
            else:
                r_wet = equilibrate_wet_radii(
                    r_dry=r_dry,
                    environment=self,
                    kappa_times_dry_volume=attributes["kappa times dry volume"],
                    rtol=rtol,
                    cell_id=attributes["cell id"],
                )
            rhod = self["rhod"].to_ndarray()
            cell_id = attributes["cell id"]
            domain_volume = np.prod(np.array(self.mesh.size))

        attributes["multiplicity"] = n_per_kg * rhod[cell_id] * domain_volume
        attributes["volume"] = self.formulae.trivia.volume(radius=r_wet)

        return attributes

    def get_thd(self):
        return (
            self.particulator.dynamics["EulerianAdvection"].solvers["th"].advectee.get()
        )

    def get_water_vapour_mixing_ratio(self):
        return (
            self.particulator.dynamics["EulerianAdvection"]
            .solvers["water_vapour_mixing_ratio"]
            .advectee.get()
        )

    def sync(self):
        self.particulator.dynamics["EulerianAdvection"].solvers.wait()
        super().sync()
