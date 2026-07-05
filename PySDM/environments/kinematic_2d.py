"""
Two-dimensional single-eddy prescribed-flow framework with moisture and heat advection
handled by [PyMPDATA](http://github.com/open-atmos/PyMPDATA/)
"""

import numpy as np

from PySDM.environments.impl.moist import Moist
from PySDM.impl.mesh import Mesh
from PySDM.initialisation.hygroscopic_equilibrium import (
    default_rtol,
    equilibrate_wet_radii,
)
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.impl import arakawa_c
from PySDM.environments.impl import register_environment


@register_environment()
class Kinematic2D(Moist):
    def __init__(self, *, dt, grid, size, rhod_of, backend, mixed_phase=False):
        super().__init__(
            dt, Mesh(grid=grid, size=size), [], mixed_phase=mixed_phase, backend=backend
        )
        self.rhod_of = rhod_of
        self.backend = backend
        self.formulae = backend.formulae
        self.dynamics = {}

        rhod = self.backend.Storage.from_ndarray(
            arakawa_c.make_rhod(self.mesh.grid, self.rhod_of).ravel()
        )
        self._values["current"]["rhod"] = rhod
        self._tmp["rhod"] = rhod

    def register_dynamics(self, dynamics):
        self.dynamics = {}
        for dynamic in dynamics:
            self.dynamics[type(dynamic).__name__] = dynamic

    def register(self, particulator):
        super().register(particulator)

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
        spectral_sampling=ConstantMultiplicity,
    ):
        super().sync()
        self.notify()
        attributes = {}

        with np.errstate(all="raise"):
            positions = spatial_discretisation.sample(
                backend=self.backend, grid=self.mesh.grid, n_sd=n_sd
            )
            (
                attributes["cell id"],
                attributes["cell origin"],
                attributes["position in cell"],
            ) = self.mesh.cellular_attributes(positions)

            r_dry, n_per_kg = spectral_sampling(
                spectrum=dry_radius_spectrum
            ).sample_deterministic(n_sd=n_sd, backend=self.backend)

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
        attributes["water mass"] = (
            self.formulae.particle_shape_and_density.radius_to_mass(r_wet)
        )

        return attributes

    def _eulerian_advection(self):
        return self.dynamics["EulerianAdvection"]

    def get_thd(self):
        return self._eulerian_advection().solvers["th"]

    def get_water_vapour_mixing_ratio(self):
        return self._eulerian_advection().solvers["water_vapour_mixing_ratio"]

    def sync(self):
        self._eulerian_advection().solvers.wait()
        super().sync()
