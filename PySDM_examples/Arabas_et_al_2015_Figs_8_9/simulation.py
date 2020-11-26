"""
Created at 25.09.2019
"""

from PySDM.backends import CPU
from PySDM.dynamics import Coalescence
from PySDM.dynamics import Condensation
from PySDM.dynamics import Displacement
from PySDM.dynamics import EulerianAdvection
from PySDM.dynamics import AmbientThermodynamics
from PySDM.products.dynamics.condensation import CondensationTimestep
from PySDM.state.arakawa_c import Fields
from PySDM.dynamics.eulerian_advection.mpdata import MPDATA
from PySDM.environments import Kinematic2D
from PySDM.products.environments import DryAirDensity
from PySDM.products.environments import DryAirPotentialTemperature
from PySDM.products.environments import RelativeHumidity, Pressure, Temperature
from PySDM.products.environments import WaterVapourMixingRatio
from PySDM.initialisation import spectral_sampling, spatial_sampling
from PySDM.builder import Builder
from PySDM.products.state import AerosolConcentration, CloudConcentration, DrizzleConcentration
from PySDM.products.state import AerosolSpecificConcentration
from PySDM.products.state import ParticleMeanRadius
from PySDM.products.state import ParticlesWetSizeSpectrum, ParticlesDrySizeSpectrum
from PySDM.products.state import SuperDropletCount
from PySDM.products.state import TotalParticleConcentration
from PySDM.products.state import TotalParticleSpecificConcentration
from PySDM.products.dynamics.displacement import SurfacePrecipitation
from PySDM.products.stats.timers import CPUTime, WallTime
from .dummy_controller import DummyController
from .spin_up import SpinUp


class Simulation:

    def __init__(self, settings, storage, backend=CPU):
        self.settings = settings
        self.storage = storage
        self.core = None
        self.backend = backend

    @property
    def products(self):
        return self.core.products

    def reinit(self, products=None):

        builder = Builder(n_sd=self.settings.n_sd, backend=self.backend)
        environment = Kinematic2D(dt=self.settings.dt,
                                  grid=self.settings.grid,
                                  size=self.settings.size,
                                  rhod_of=self.settings.rhod,
                                  field_values=self.settings.field_values)
        builder.set_environment(environment)

        products = products or [
            ParticlesWetSizeSpectrum(v_bins=self.settings.v_bins, normalise_by_dv=True),
            ParticlesDrySizeSpectrum(v_bins=self.settings.v_bins, normalise_by_dv=True),  # Note: better v_bins
            TotalParticleConcentration(),
            TotalParticleSpecificConcentration(),
            AerosolConcentration(radius_threshold=self.settings.aerosol_radius_threshold),
            CloudConcentration(radius_range=(self.settings.aerosol_radius_threshold, self.settings.drizzle_radius_threshold)),
            DrizzleConcentration(radius_threshold=self.settings.drizzle_radius_threshold),
            AerosolSpecificConcentration(radius_threshold=self.settings.aerosol_radius_threshold),
            ParticleMeanRadius(),
            SuperDropletCount(),
            RelativeHumidity(), Pressure(), Temperature(),
            WaterVapourMixingRatio(),
            DryAirDensity(),
            DryAirPotentialTemperature(),
            CPUTime(),
            WallTime()
        ]

        fields = Fields(environment, self.settings.stream_function)
        if self.settings.processes['fluid advection']:  # TODO: ambient thermodynamics checkbox
            builder.add_dynamic(AmbientThermodynamics())
        if self.settings.processes["condensation"]:
            condensation = Condensation(
                kappa=self.settings.kappa,
                rtol_x=self.settings.condensation_rtol_x,
                rtol_thd=self.settings.condensation_rtol_thd,
                coord=self.settings.condensation_coord,
                adaptive=self.settings.adaptive)
            builder.add_dynamic(condensation)
            products.append(CondensationTimestep())  # TODO: and what if a user doesn't want it?
        if self.settings.processes['fluid advection']:
            solver = MPDATA(
                fields=fields,
                n_iters=self.settings.mpdata_iters,
                infinite_gauge=self.settings.mpdata_iga,
                flux_corrected_transport=self.settings.mpdata_fct,
                third_order_terms=self.settings.mpdata_tot
            )
            builder.add_dynamic(EulerianAdvection(solver))
        if self.settings.processes["particle advection"]:
            displacement = Displacement(
                courant_field=fields.courant_field,
                scheme='FTBS',
                enable_sedimentation=self.settings.processes["sedimentation"])
            builder.add_dynamic(displacement)
            products.append(SurfacePrecipitation())  # TODO: ditto
        if self.settings.processes["coalescence"]:
            builder.add_dynamic(Coalescence(kernel=self.settings.kernel))

        attributes = environment.init_attributes(spatial_discretisation=spatial_sampling.Pseudorandom(),
                                                 spectral_discretisation=spectral_sampling.ConstantMultiplicity(
                                                     spectrum=self.settings.spectrum_per_mass_of_dry_air
                                                 ),
                                                 kappa=self.settings.kappa)

        self.core = builder.build(attributes, products)
        SpinUp(self.core, self.settings.n_spin_up)
        # TODO
        if self.storage is not None:
            self.storage.init(self.settings)

    def run(self, controller=DummyController()):
        with controller:
            for step in self.settings.steps:  # TODO: rename output_steps
                if controller.panic:
                    break

                self.core.run(step - self.core.n_steps)

                self.store(step)

                controller.set_percent(step / self.settings.steps[-1])

    def store(self, step):
        for name, product in self.core.products.items():
            self.storage.save(product.get(), step, name)
