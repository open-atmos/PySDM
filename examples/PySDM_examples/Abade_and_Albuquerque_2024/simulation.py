import numpy as np

from PySDM_examples.utils import BasicSimulation

from PySDM import Builder
from PySDM.backends import CPU
from PySDM.dynamics import Condensation, AmbientThermodynamics
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.products import AmbientTemperature, ParcelDisplacement, WaterMixingRatio
from PySDM.environments import Parcel


class Simulation(BasicSimulation):
    def __init__(self, settings):
        parcel = Parcel(
            dt=settings.timestep,
            mass_of_dry_air=settings.mass_of_dry_air,
            p0=settings.initial_total_pressure,
            initial_water_vapour_mixing_ratio=settings.initial_water_vapour_mixing_ratio,
            T0=settings.initial_temperature,
            w=settings.updraft,
        )
        builder = Builder(
            backend=CPU(settings.formulae, override_jit_flags={"parallel": False}),
            n_sd=settings.n_sd,
            environment=parcel,
        )
        builder.add_dynamic(AmbientThermodynamics())
        builder.add_dynamic(Condensation())

        r_dry, n_in_dv = ConstantMultiplicity(settings.soluble_aerosol).sample(
            n_sd=settings.n_sd
        )
        attributes = parcel.init_attributes(
            n_in_dv=n_in_dv, kappa=settings.kappa, r_dry=r_dry
        )
        self.products = (
            WaterMixingRatio(name="water", radius_range=(0, np.inf)),
            WaterMixingRatio(name="ice", radius_range=(-np.inf, 0)),
            WaterMixingRatio(name="total", radius_range=(-np.inf, np.inf)),
            ParcelDisplacement(name="height"),
            AmbientTemperature(name="T"),
        )
        super().__init__(
            particulator=builder.build(attributes=attributes, products=self.products)
        )

    def run(self, *, nt, steps_per_output_interval):
        return self._run(nt=nt, steps_per_output_interval=steps_per_output_interval)
