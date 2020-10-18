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
from .dummy_controller import DummyController
from .spin_up import SpinUp


class Simulation:

    def __init__(self, setup, storage, backend=CPU):
        self.setup = setup
        self.storage = storage
        self.core = None
        self.backend = backend

    @property
    def products(self):
        return self.core.products

    def reinit(self):

        builder = Builder(n_sd=self.setup.n_sd, backend=self.backend)
        environment = Kinematic2D(dt=self.setup.dt,
                                  grid=self.setup.grid,
                                  size=self.setup.size,
                                  rhod_of=self.setup.rhod,
                                  field_values=self.setup.field_values)
        builder.set_environment(environment)

        products = [
            ParticlesWetSizeSpectrum(v_bins=self.setup.v_bins, normalise_by_dv=True),
            ParticlesDrySizeSpectrum(v_bins=self.setup.v_bins, normalise_by_dv=True),  # Note: better v_bins
            TotalParticleConcentration(),
            TotalParticleSpecificConcentration(),
            AerosolConcentration(radius_threshold=self.setup.aerosol_radius_threshold),
            CloudConcentration(radius_range=(self.setup.aerosol_radius_threshold, self.setup.drizzle_radius_threshold)),
            DrizzleConcentration(radius_threshold=self.setup.drizzle_radius_threshold),
            AerosolSpecificConcentration(radius_threshold=self.setup.aerosol_radius_threshold),
            ParticleMeanRadius(),
            SuperDropletCount(),
            RelativeHumidity(), Pressure(), Temperature(),
            WaterVapourMixingRatio(),
            DryAirDensity(),
            DryAirPotentialTemperature()
        ]

        fields = Fields(environment, self.setup.stream_function)
        if self.setup.processes['fluid advection']:  # TODO: ambient thermodynamics checkbox
            builder.add_dynamic(AmbientThermodynamics())
        if self.setup.processes["condensation"]:
            condensation = Condensation(
                kappa=self.setup.kappa,
                rtol_x=self.setup.condensation_rtol_x,
                rtol_thd=self.setup.condensation_rtol_thd,
                coord=self.setup.condensation_coord,
                adaptive=self.setup.adaptive)
            builder.add_dynamic(condensation)
            products.append(CondensationTimestep())
        if self.setup.processes['fluid advection']:
            mpdatas = MPDATA(fields=fields,
                             n_iters=self.setup.mpdata_iters,
                             infinite_gauge=self.setup.mpdata_iga,
                             flux_corrected_transport=self.setup.mpdata_fct,
                             third_order_terms=self.setup.mpdata_tot)
            builder.add_dynamic(EulerianAdvection(mpdatas))
        if self.setup.processes["particle advection"]:
            displacement = Displacement(
                courant_field=fields.courant_field,
                scheme='FTBS',
                enable_sedimentation=self.setup.processes["sedimentation"])
            builder.add_dynamic(displacement)
        if self.setup.processes["coalescence"]:
            builder.add_dynamic(Coalescence(kernel=self.setup.kernel))

        attributes = environment.init_attributes(spatial_discretisation=spatial_sampling.Pseudorandom(),
                                                 spectral_discretisation=spectral_sampling.ConstantMultiplicity(
                                                     spectrum=self.setup.spectrum_per_mass_of_dry_air
                                                 ),
                                                 kappa=self.setup.kappa)

        self.core = builder.build(attributes, products)
        SpinUp(self.core, self.setup.n_spin_up)
        # TODO
        if self.storage is not None:
            self.storage.init(self.setup)

    def run(self, controller=DummyController()):
        with controller:
            for step in self.setup.steps:  # TODO: rename output_steps
                if controller.panic:
                    break

                self.core.run(step - self.core.n_steps)

                self.store(step)

                controller.set_percent(step / self.setup.steps[-1])

        return self.core.stats

    def store(self, step):
        for name, product in self.core.products.items():
            self.storage.save(product.get(), step, name)
