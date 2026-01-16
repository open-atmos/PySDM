import numpy as np
from PySDM_examples.utils import BasicSimulation

from PySDM.backends import CPU
from PySDM import Builder
from PySDM.dynamics import Coalescence
from PySDM.environments import Box
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.products.size_spectral import ParticleVolumeVersusRadiusLogarithmSpectrum
from PySDM.products import Time

class Simulation(BasicSimulation):
    def __init__(self, settings, products=None):
        builder = Builder(
            n_sd=settings.n_sd,
            backend=CPU(settings.formulae),
            environment=Box(dv=settings.dv, dt=settings.dt),
        )
        builder.particulator.environment["rhod"] = settings.rhod
        attributes = {}
        attributes["volume"], attributes["multiplicity"] = ConstantMultiplicity(
            settings.spectrum
        ).sample_deterministic(settings.n_sd)
        builder.add_dynamic(
            Coalescence(
                collision_kernel=settings.kernel,
                coalescence_efficiency=settings.coal_eff,
                adaptive=settings.adaptive,
            )
        )
        products = (
            ParticleVolumeVersusRadiusLogarithmSpectrum(
                radius_bins_edges=settings.radius_bins_edges, name="dv/dlnr"
            ),
            Time(name="t"),
        )
        particulator = builder.build(attributes, products)
        self.settings = settings
        super().__init__(particulator=particulator)


    def run(self):
        return super()._run(self.settings.nt, self.settings.steps_per_output_interval)
