"""
Created at 25.09.2019
"""

from PySDM.backends import CPU
from PySDM.builder import Builder
from PySDM.dynamics import AmbientThermodynamics
from PySDM.dynamics import Coalescence
from PySDM.dynamics import Condensation
from PySDM.dynamics import Displacement
from PySDM.dynamics import EulerianAdvection
from PySDM.dynamics.eulerian_advection.mpdata import MPDATA
from PySDM.environments import Kinematic2D
from PySDM.initialisation import spectral_sampling, spatial_sampling
from PySDM import products as PySDM_products
from PySDM.state.arakawa_c import Fields
from .dummy_controller import DummyController
from .spin_up import SpinUp
import numpy as np


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

        cloud_range = (self.settings.aerosol_radius_threshold, self.settings.drizzle_radius_threshold)
        if products is not None:
            products = list(products)
        products = products or [
            PySDM_products.ParticlesWetSizeSpectrum(v_bins=self.settings.v_bins, normalise_by_dv=True),
            PySDM_products.ParticlesDrySizeSpectrum(v_bins=self.settings.v_bins, normalise_by_dv=True),  # Note: better v_bins
            PySDM_products.TotalParticleConcentration(),
            PySDM_products.TotalParticleSpecificConcentration(),
            PySDM_products.AerosolConcentration(radius_threshold=self.settings.aerosol_radius_threshold),
            PySDM_products.CloudDropletConcentration(radius_range=cloud_range),
            PySDM_products.WaterMixingRatio(name='qc', description_prefix='Cloud', radius_range=cloud_range),
            PySDM_products.WaterMixingRatio(name='qr', description_prefix='Rain', radius_range=(self.settings.drizzle_radius_threshold, np.inf)),
            PySDM_products.DrizzleConcentration(radius_threshold=self.settings.drizzle_radius_threshold),
            PySDM_products.AerosolSpecificConcentration(radius_threshold=self.settings.aerosol_radius_threshold),
            PySDM_products.ParticleMeanRadius(),
            PySDM_products.SuperDropletCount(),
            PySDM_products.RelativeHumidity(), PySDM_products.Pressure(), PySDM_products.Temperature(),
            PySDM_products.WaterVapourMixingRatio(),
            PySDM_products.DryAirDensity(),
            PySDM_products.DryAirPotentialTemperature(),
            PySDM_products.CPUTime(),
            PySDM_products.WallTime(),
            PySDM_products.CloudDropletEffectiveRadius(radius_range=cloud_range),
            PySDM_products.PeakSupersaturation(),
            PySDM_products.ActivatingRate(),
            PySDM_products.DeactivatingRate(),
            PySDM_products.RipeningRate()
        ]

        fields = Fields(environment, self.settings.stream_function)
        if self.settings.processes['fluid advection']:  # TODO #37 ambient thermodynamics checkbox
            builder.add_dynamic(AmbientThermodynamics())
        if self.settings.processes["condensation"]:
            condensation = Condensation(
                kappa=self.settings.kappa,
                rtol_x=self.settings.condensation_rtol_x,
                rtol_thd=self.settings.condensation_rtol_thd,
                coord=self.settings.condensation_coord,
                adaptive=self.settings.condensation_adaptive,
                substeps=self.settings.condensation_substeps,
                dt_cond_range=self.settings.condensation_dt_cond_range,
                schedule=self.settings.condensation_schedule
            )
            builder.add_dynamic(condensation)
            products.append(PySDM_products.CondensationTimestepMin())  # TODO #37 and what if a user doesn't want it?
            products.append(PySDM_products.CondensationTimestepMax())
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
            products.append(PySDM_products.SurfacePrecipitation())  # TODO #37 ditto
        if self.settings.processes["coalescence"]:
            builder.add_dynamic(Coalescence(
                kernel=self.settings.kernel,
                adaptive=self.settings.coalescence_adaptive,
                dt_coal_range=self.settings.coalescence_dt_coal_range,
                substeps=self.settings.coalescence_substeps,
                optimized_random=self.settings.coalescence_optimized_random
            ))
            products.append(PySDM_products.CoalescenceTimestepMean())
            products.append(PySDM_products.CoalescenceTimestepMin())
            products.append(PySDM_products.CollisionRate())
            products.append(PySDM_products.CollisionRateDeficit())

        attributes = environment.init_attributes(spatial_discretisation=spatial_sampling.Pseudorandom(),
                                                 spectral_discretisation=spectral_sampling.ConstantMultiplicity(
                                                     spectrum=self.settings.spectrum_per_mass_of_dry_air
                                                 ),
                                                 kappa=self.settings.kappa)

        self.core = builder.build(attributes, products)
        SpinUp(self.core, self.settings.n_spin_up)
        if self.storage is not None:
            self.storage.init(self.settings)

    def run(self, controller=DummyController()):
        with controller:
            for step in self.settings.output_steps:
                if controller.panic:
                    break

                self.core.run(step - self.core.n_steps)

                self.store(step)

                controller.set_percent(step / self.settings.output_steps[-1])

    def store(self, step):
        for name, product in self.core.products.items():
            self.storage.save(product.get(), step, name)
