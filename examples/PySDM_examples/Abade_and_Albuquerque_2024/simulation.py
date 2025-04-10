import numpy as np

from PySDM_examples.utils import BasicSimulation

from PySDM import Builder
from PySDM.backends import CPU
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


class Simulation(BasicSimulation):
    def __init__(self, settings):
        builder = Builder(
            backend=CPU(settings.formulae, override_jit_flags={"parallel": False}),
            n_sd=settings.n_sd,
            environment=Parcel(
                dt=settings.timestep,
                mass_of_dry_air=settings.mass_of_dry_air,
                p0=settings.initial_total_pressure,
                initial_water_vapour_mixing_ratio=settings.initial_water_vapour_mixing_ratio,
                T0=settings.initial_temperature,
                w=settings.updraft,
                mixed_phase=True,
            ),
        )
        builder.add_dynamic(AmbientThermodynamics())
        builder.add_dynamic(Condensation())

        if settings.enable_immersion_freezing:
            builder.add_dynamic(Freezing())
        if settings.enable_vapour_deposition_on_ice:
            builder.add_dynamic(VapourDepositionOnIce())

        r_dry, n_in_dv = ConstantMultiplicity(settings.soluble_aerosol).sample(
            n_sd=settings.n_sd
        )
        attributes = builder.particulator.environment.init_attributes(
            n_in_dv=n_in_dv, kappa=settings.kappa, r_dry=r_dry
        )
        attributes["signed water mass"] = (
            builder.particulator.formulae.particle_shape_and_density.volume_to_mass(
                attributes["volume"]
            )
        )
        del attributes["volume"]

        if settings.enable_immersion_freezing:
            trivia = builder.particulator.formulae.trivia
            n_inp = int(settings.n_sd * settings.freezing_inp_frac)

            rng = np.random.default_rng(seed=builder.particulator.formulae.seed)

            attributes["freezing temperature"] = rng.permutation(
                np.pad(
                    builder.particulator.formulae.freezing_temperature_spectrum.invcdf(
                        cdf=rng.uniform(low=0, high=1, size=n_inp),
                        A_insol=trivia.sphere_surface(
                            diameter=2 * settings.freezing_inp_dry_radius
                        ),
                    ),
                    (0, settings.n_sd - n_inp),
                    mode="constant",
                    constant_values=(trivia.C2K(-38)),
                )
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
        super().__init__(
            particulator=builder.build(attributes=attributes, products=self.products)
        )

    def run(self, *, nt, steps_per_output_interval):
        return self._run(nt=nt, steps_per_output_interval=steps_per_output_interval)
