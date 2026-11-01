import numpy as np

from PySDM_examples.utils import BasicSimulation

from PySDM.particulator import Particulator
from PySDM.dynamics import (
    Condensation,
    AmbientThermodynamics,
    VapourDepositionOnIce,
    Freezing,
)
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.products import (
    AmbientTemperature,
    AmbientWaterVapourMixingRatio,
    ParcelDisplacement,
    WaterMixingRatio,
    SpecificIceWaterContent,
)
from PySDM.environments import Parcel
from PySDM_examples.sandbox.Abade_and_Albuquerque_2024.sampling import sampling


class Simulation(BasicSimulation):
    def __init__(self, settings):

        dynamics = [
            AmbientThermodynamics(),
            Condensation(),
        ]

        if settings.enable_immersion_freezing:
            dynamics.append(
                Freezing(
                    immersion_freezing=(
                        "singular" if settings.singular else "time-dependent"
                    )
                )
            )

        if settings.enable_vapour_deposition_on_ice:
            dynamics.append(VapourDepositionOnIce(adaptive=True))

        environment = Parcel(
            dt=settings.timestep,
            mass_of_dry_air=settings.mass_of_dry_air,
            p0=settings.initial_total_pressure,
            initial_water_vapour_mixing_ratio=settings.initial_water_vapour_mixing_ratio,
            T0=settings.initial_temperature,
            w=settings.updraft,
            mixed_phase=True,
            backend=settings.backend,
        )
        r_dry, n_in_dv = ConstantMultiplicity(
            settings.soluble_aerosol
        ).sample_deterministic(n_sd=settings.n_sd)
        attributes = environment.init_attributes(
            n_in_dv=n_in_dv,
            kappa=settings.kappa,
            r_dry=r_dry,
        )
        attributes["signed water mass"] = (
            settings.backend.formulae.particle_shape_and_density.volume_to_mass(
                attributes["volume"]
            )
        )
        del attributes["volume"]

        if settings.enable_immersion_freezing:

            rng = np.random.default_rng(seed=settings.backend.formulae.seed)

            sampling(
                n_sd=settings.n_sd,
                freezing_inp_frac=settings.freezing_inp_frac,
                rng=rng,
                singular=settings.singular,
                freezing_inp_dry_radius=settings.freezing_inp_dry_radius,
                attributes=attributes,
                formulae=settings.backend.formulae,
            )

        self.products = (
            WaterMixingRatio(name="water", radius_range=(0, np.inf)),
            SpecificIceWaterContent(name="ice"),
            ParcelDisplacement(name="height"),
            AmbientTemperature(name="T"),
            AmbientWaterVapourMixingRatio(
                name="vapour", var="water_vapour_mixing_ratio"
            ),
        )
        particulator = Particulator(
            n_sd=settings.n_sd,
            environment=environment,
            dynamics=dynamics,
            attributes=attributes,
            products=self.products,
        )
        super().__init__(particulator=particulator)

    def run(self, *, nt, steps_per_output_interval):
        return self._run(nt=nt, steps_per_output_interval=steps_per_output_interval)
