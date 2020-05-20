"""
Created at 25.09.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import time

from PySDM.particles_builder import ParticlesBuilder
from PySDM.dynamics import Displacement
from PySDM.dynamics import Condensation
from PySDM.dynamics import EulerianAdvection
from PySDM.dynamics import Coalescence
from PySDM.initialisation import spectral_sampling, spatial_sampling
from PySDM.environments import MoistEulerian2DKinematic
from PySDM.initialisation.moist_environment_init import moist_environment_init

from PySDM.state.products.aerosol_concentration import AerosolConcentration
from PySDM.state.products.aerosol_specific_concentration import AerosolSpecificConcentration
from PySDM.state.products.total_particle_concentration import TotalParticleConcentration
from PySDM.state.products.total_particle_specific_concentration import TotalParticleSpecificConcentration
from PySDM.state.products.particle_mean_radius import ParticleMeanRadius
from PySDM.state.products.super_droplet_count import SuperDropletCount
from PySDM.state.products.particle_temperature import ParticleTemperature
from PySDM.environments.products.relative_humidity import RelativeHumidity
from PySDM.environments.products.dry_air_potential_temperature import DryAirPotentialTemperature
from PySDM.environments.products.water_vapour_mixing_ratio import WaterVapourMixingRatio
from PySDM.environments.products.dry_air_density import DryAirDensity
from PySDM.dynamics.condensation.products.condensation_timestep import CondensationTimestep
from PySDM.dynamics.condensation.products.ripening_flag import RipeningFlag


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
        self.particles = None

    @property
    def products(self):
        return self.particles.products

    def reinit(self):

        particles_builder = ParticlesBuilder(n_sd=self.setup.n_sd, backend=self.setup.backend)
        particles_builder.set_environment(MoistEulerian2DKinematic, {
            "dt": self.setup.dt,
            "grid": self.setup.grid,
            "size": self.setup.size,
            "stream_function": self.setup.stream_function,
            "field_values": self.setup.field_values,
            "rhod_of": self.setup.rhod,
            "mpdata_iga": self.setup.mpdata_iga,
            "mpdata_tot": self.setup.mpdata_tot,
            "mpdata_fct": self.setup.mpdata_fct,
            "mpdata_iters": self.setup.mpdata_iters
        })


        particles_builder.register_dynamic(Condensation, {
            "kappa": self.setup.kappa,
            "rtol_x": self.setup.condensation_rtol_x,
            "rtol_thd": self.setup.condensation_rtol_thd,
            "coord": self.setup.condensation_coord,
            "do_advection": self.setup.processes["fluid advection"],  # TODO req. EulerianAdvection
            "do_condensation": self.setup.processes["condensation"]  # do somthing with that
        })
        particles_builder.register_dynamic(EulerianAdvection, {})

        if self.setup.processes["particle advection"]:
            particles_builder.register_dynamic(
                Displacement, {"scheme": 'FTBS', "sedimentation": self.setup.processes["sedimentation"]})
        if self.setup.processes["coalescence"]:
            particles_builder.register_dynamic(Coalescence, {"kernel": self.setup.kernel})
        # TODO
        # if self.setup.processes["relaxation"]:
        #     raise NotImplementedError()

        attributes = {}
        moist_environment_init(attributes, particles_builder.particles.environment,
                               spatial_discretisation=spatial_sampling.pseudorandom,
                               spectral_discretisation=spectral_sampling.constant_multiplicity,  # TODO: random
                               spectrum_per_mass_of_dry_air=self.setup.spectrum_per_mass_of_dry_air,
                               r_range=(self.setup.r_min, self.setup.r_max),
                               kappa=self.setup.kappa)
        products = {
            TotalParticleConcentration: {},
            TotalParticleSpecificConcentration: {},
            AerosolConcentration: {'radius_threshold': self.setup.aerosol_radius_threshold},
            AerosolSpecificConcentration: {'radius_threshold': self.setup.aerosol_radius_threshold},
            ParticleMeanRadius: {},
            SuperDropletCount: {},
            RelativeHumidity: {},
            WaterVapourMixingRatio: {},
            DryAirDensity: {},
            DryAirPotentialTemperature: {},
            CondensationTimestep: {},
            RipeningFlag: {}
        }
        self.particles = particles_builder.get_particles(attributes, products)

        # TODO
        if self.storage is not None:
            self.storage.init(self.setup)

    def run(self, controller=DummyController()):
        with controller:
            for step in self.setup.steps:  # TODO: rename output_steps
                if controller.panic:
                    break

                self.particles.run(step - self.particles.n_steps)

                self.store(step)

                controller.set_percent(step / self.setup.steps[-1])

        return self.particles.stats

    def store(self, step):
        for name, product in self.particles.products.items():
            self.storage.save(product.get(), step, name)
