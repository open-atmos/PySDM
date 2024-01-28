import numpy as np
from PySDM_examples.Ervens_and_Feingold_2012.settings import (
    sampled_ccn_diameter_number_concentration_spectrum,
)
from PySDM_examples.Niedermeier_et_al_2014.settings import Settings
from PySDM_examples.utils import BasicSimulation

from PySDM import Builder
from PySDM.backends import CPU
from PySDM.dynamics import AmbientThermodynamics, Condensation, Freezing
from PySDM.environments import Parcel
from PySDM.initialisation import equilibrate_wet_radii
from PySDM.products import AmbientTemperature, IceWaterContent, ParcelDisplacement


class Simulation(BasicSimulation):
    def __init__(self, settings: Settings):
        n_particles = settings.ccn_sampling_n - 1 + settings.in_sampling_n
        env = Parcel(
            dt=settings.timestep,
            p0=settings.p0,
            T0=settings.T0,
            initial_water_vapour_mixing_ratio=settings.initial_water_vapour_mixing_ratio,
            mass_of_dry_air=settings.mass_of_dry_air,
            w=settings.vertical_velocity,
            mixed_phase=True,
        )
        builder = Builder(
            n_sd=n_particles, backend=CPU(settings.formulae), environment=env
        )

        builder.add_dynamic(AmbientThermodynamics())
        builder.add_dynamic(Condensation())
        builder.add_dynamic(Freezing(singular=False))

        air_volume = settings.mass_of_dry_air / settings.rhod0
        (
            ccn_diameter,
            ccn_conc_float,
        ) = sampled_ccn_diameter_number_concentration_spectrum(
            size_range=settings.ccn_dry_diameter_range, n_sd=settings.ccn_sampling_n
        )
        dry_volume = settings.formulae.trivia.volume(radius=ccn_diameter / 2)

        immersed_surface_area = np.zeros_like(dry_volume)
        immersed_surface_area[-1] = settings.formulae.trivia.sphere_surface(
            diameter=ccn_diameter[-1]
        )

        attributes = {
            "multiplicity": ccn_conc_float * air_volume,
            "dry volume": dry_volume,
            "kappa times dry volume": settings.kappa * dry_volume,
            "volume": None,
            "immersed surface area": immersed_surface_area,
        }
        attributes["volume"] = settings.formulae.trivia.volume(
            radius=equilibrate_wet_radii(
                r_dry=ccn_diameter / 2,
                environment=builder.particulator.environment,
                kappa_times_dry_volume=attributes["kappa times dry volume"],
            )
        )

        for attribute, data in attributes.items():
            attributes[attribute] = np.concatenate(
                (
                    data[:-1],
                    np.full(
                        settings.in_sampling_n,
                        (
                            data[-1]
                            if attribute != "multiplicity"
                            else data[-1] / settings.in_sampling_n
                        ),
                    ),
                )
            )

        products = (
            IceWaterContent(),
            ParcelDisplacement(name="z"),
            AmbientTemperature(name="T"),
        )
        super().__init__(builder.build(attributes=attributes, products=products))
        self.steps = int(
            settings.displacement / settings.vertical_velocity / settings.timestep
        )

    def run(self):
        return super()._run(nt=self.steps, steps_per_output_interval=1)
