"""
Created at 25.09.2019
"""


import time

from PySDM.dynamics import Coalescence
from PySDM.dynamics import Condensation
from PySDM.dynamics import Displacement
from PySDM.dynamics import EulerianAdvection
from PySDM.dynamics.condensation.products.condensation_timestep import CondensationTimestep
from PySDM.dynamics.condensation.products.ripening_rate import RipeningRate
from PySDM.environments import MoistEulerian2DKinematic
from PySDM.environments.products.dry_air_density import DryAirDensity
from PySDM.environments.products.dry_air_potential_temperature import DryAirPotentialTemperature
from PySDM.environments.products.relative_humidity import RelativeHumidity
from PySDM.environments.products.water_vapour_mixing_ratio import WaterVapourMixingRatio
from PySDM.initialisation import spectral_sampling, spatial_sampling
from PySDM.initialisation.moist_environment_init import moist_environment_init
from PySDM.builder import Builder
from PySDM.state.products.particles_concentration import AerosolConcentration, CloudConcentration, DrizzleConcentration
from PySDM.state.products.aerosol_specific_concentration import AerosolSpecificConcentration
from PySDM.state.products.particle_mean_radius import ParticleMeanRadius
from PySDM.state.products.particles_size_spectrum import ParticlesWetSizeSpectrum, ParticlesDrySizeSpectrum
from PySDM.state.products.super_droplet_count import SuperDropletCount
from PySDM.state.products.total_particle_concentration import TotalParticleConcentration
from PySDM.state.products.total_particle_specific_concentration import TotalParticleSpecificConcentration
from .spin_up import SpinUp


class DummyController:

    def __init__(self):
        self.panic = False
        self.t_last = self.__times()

    def __times(self):
        return time.perf_counter(), time.process_time()

    def set_percent(self, value):
        t_curr = self.__times()
        wall_time = (t_curr[0] - self.t_last[0])
        cpu_time = (t_curr[1] - self.t_last[1])
        print(f"{100 * value:.1f}% (times since last print: cpu={cpu_time:.1f}s wall={wall_time:.1f}s)")
        self.t_last = self.__times()

    def __enter__(*_): pass

    def __exit__(*_): pass


class Simulation:

    def __init__(self, setup, storage):
        self.setup = setup
        self.storage = storage
        self.core = None

    @property
    def products(self):
        return self.core.products

    def reinit(self):

        builder = Builder(n_sd=self.setup.n_sd, backend=self.setup.backend)
        builder.set_environment(MoistEulerian2DKinematic(
            dt=self.setup.dt,
            grid=self.setup.grid,
            size=self.setup.size,
            stream_function=self.setup.stream_function,
            field_values=self.setup.field_values,
            rhod_of=self.setup.rhod,
            mpdata_iga=self.setup.mpdata_iga,
            mpdata_tot=self.setup.mpdata_tot,
            mpdata_fct=self.setup.mpdata_fct,
            mpdata_iters=self.setup.mpdata_iters
        ))

        condensation = Condensation(
            kappa=self.setup.kappa,
            rtol_x=self.setup.condensation_rtol_x,
            rtol_thd=self.setup.condensation_rtol_thd,
            coord=self.setup.condensation_coord,
            adaptive=self.setup.adaptive,
            do_advection=self.setup.processes["fluid advection"],  # TODO req. EulerianAdvection
            do_condensation=self.setup.processes["condensation"])  # do somthing with that)
        builder.add_dynamic(condensation)
        builder.add_dynamic(EulerianAdvection())

        if self.setup.processes["particle advection"]:
            displacement = Displacement(scheme='FTBS', sedimentation=self.setup.processes["sedimentation"])
            builder.add_dynamic(displacement)
        if self.setup.processes["coalescence"]:
            builder.add_dynamic(Coalescence(kernel=self.setup.kernel))

        attributes = {}
        moist_environment_init(attributes, builder.core.environment,
                               spatial_discretisation=spatial_sampling.pseudorandom,
                               spectral_discretisation=spectral_sampling.constant_multiplicity,
                               spectrum_per_mass_of_dry_air=self.setup.spectrum_per_mass_of_dry_air,
                               r_range=(self.setup.r_min, self.setup.r_max),
                               kappa=self.setup.kappa)
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
            RelativeHumidity(),
            WaterVapourMixingRatio(),
            DryAirDensity(),
            DryAirPotentialTemperature(),
            CondensationTimestep(),
            # RipeningRate()
        ]
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
