from PySDM_examples.utils import BasicSimulation

from PySDM.backends import CPU
from PySDM import Particulator
from PySDM.dynamics import Coalescence
from PySDM.environments import Box
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.products.size_spectral import ParticleVolumeVersusRadiusLogarithmSpectrum
from PySDM.products import Time


class Simulation(BasicSimulation):
    def __init__(self, settings, products=None):
        environment = Box(dv=settings.dv, dt=settings.dt, backend=CPU())
        environment["rhod"] = settings.rhod
        attributes = {}
        attributes["volume"], attributes["multiplicity"] = ConstantMultiplicity(
            settings.spectrum
        ).sample_deterministic(settings.n_sd)

        products = (
            ParticleVolumeVersusRadiusLogarithmSpectrum(
                radius_bins_edges=settings.radius_bins_edges, name="dv/dlnr"
            ),
            Time(name="t"),
        )
        particulator = Particulator(
            n_sd=settings.n_sd,
            environment=environment,
            dynamics=(
                Coalescence(
                    collision_kernel=settings.kernel,
                    coalescence_efficiency=settings.coal_eff,
                    adaptive=settings.adaptive,
                ),
            ),
            attributes=attributes,
            products=products,
        )
        self.settings = settings
        super().__init__(particulator=particulator)

    def run(self):
        return super()._run(self.settings.nt, self.settings.steps_per_output_interval)
