import numpy as np

from PySDM_examples.utils import BasicSimulation

import PySDM.products as PySDM_products
from PySDM.backends import CPU
from PySDM.builder import Builder
from PySDM.dynamics import (
    AmbientThermodynamics,
    Condensation,
    Freezing,
    VapourDepositionOnIce,
)
from PySDM.environments import Parcel
from PySDM.initialisation import discretise_multiplicities, equilibrate_wet_radii


class Simulation(BasicSimulation):
    def __init__(self, settings, backend=CPU):

        dt = settings.dt

        formulae = settings.formulae

        env = Parcel(
            mixed_phase=True,
            dt=dt,
            mass_of_dry_air=settings.mass_of_dry_air,
            p0=settings.initial_pressure,
            initial_water_vapour_mixing_ratio=settings.initial_water_vapour_mixing_ratio,
            T0=settings.initial_temperature,
            w=settings.w_updraft,
        )

        builder = Builder(
            backend=backend(
                formulae=settings.formulae,
                **(
                    {"override_jit_flags": {"parallel": False}}
                    if backend == CPU
                    else {}
                ),
            ),
            n_sd=settings.n_sd,
            environment=env,
        )

        builder.add_dynamic(AmbientThermodynamics())
        builder.add_dynamic(Condensation())
        builder.add_dynamic(VapourDepositionOnIce())
        builder.add_dynamic(
            Freezing(
                singular=False, homogeneous_freezing=True, immersion_freezing=False
            )
        )

        self.n_sd = settings.n_sd
        self.multiplicities = discretise_multiplicities(
            settings.specific_concentration * env.mass_of_dry_air
        )
        self.r_dry = settings.r_dry
        v_dry = settings.formulae.trivia.volume(radius=self.r_dry)
        kappa = settings.kappa

        self.r_wet = equilibrate_wet_radii(
            r_dry=self.r_dry,
            environment=builder.particulator.environment,
            kappa_times_dry_volume=kappa * v_dry,
        )

        attributes = {
            "multiplicity": self.multiplicities,
            "dry volume": v_dry,
            "kappa times dry volume": kappa * v_dry,
            "signed water mass": formulae.particle_shape_and_density.radius_to_mass(
                self.r_wet
            ),
        }

        products = [
            PySDM_products.Time(name="t"),
            PySDM_products.AmbientRelativeHumidity(name="RH_ice", unit="%"),
            PySDM_products.ParticleConcentration(
                name="n_i", unit="1/m**3", radius_range=(-np.inf, 0)
            ),
        ]

        self.n_output = settings.n_output
        self.n_substeps = int(settings.t_duration / dt / self.n_output)
        super().__init__(builder.build(attributes, products))

    def save(self, output):
        cell_id = 0
        output["t"].append(self.particulator.products["t"].get())
        output["ni"].append(self.particulator.products["n_i"].get()[cell_id])
        output["RHi"].append(self.particulator.products["RH_ice"].get()[cell_id])

    def run(self):
        output = {
            "t": [],
            "ni": [],
            "RHi": [],
        }

        self.save(output)

        RHi_old = self.particulator.products["RH_ice"].get()[0].copy()
        for _ in range(self.n_output):

            self.particulator.run(self.n_substeps)

            self.save(output)

            RHi = self.particulator.products["RH_ice"].get()[0].copy()
            dRHi = (RHi_old - RHi) / RHi_old
            if dRHi > 0.0 and RHi < 130.0:
                print("break")
                break
            RHi_old = RHi

        return output["ni"][-1]
