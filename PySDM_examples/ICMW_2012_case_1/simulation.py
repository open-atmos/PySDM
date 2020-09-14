"""
Created at 25.09.2019
"""


import time
import numpy as np

from PySDM.dynamics import Coalescence
from PySDM.dynamics import Condensation
from PySDM.dynamics import Displacement
from PySDM.dynamics import EulerianAdvection
from PySDM.dynamics import AmbientThermodynamics
from PySDM.products.dynamics.condensation import CondensationTimestep
from PySDM.environments.kinematic_2d.arakawa_c import make_rhod, nondivergent_vector_field_2d
from PySDM.environments.kinematic_2d.mpdata import MPDATA
from PySDM.environments import MoistEulerian2DKinematic
from PySDM.products.environments import DryAirDensity
from PySDM.products.environments import DryAirPotentialTemperature
from PySDM.products.environments import RelativeHumidity
from PySDM.products.environments import WaterVapourMixingRatio
from PySDM.initialisation import spectral_sampling, spatial_sampling
from PySDM.initialisation.moist_environment_init import moist_environment_init
from PySDM.builder import Builder
from PySDM.products.state import AerosolConcentration, CloudConcentration, DrizzleConcentration
from PySDM.products.state import AerosolSpecificConcentration
from PySDM.products.state import ParticleMeanRadius
from PySDM.products.state import ParticlesWetSizeSpectrum, ParticlesDrySizeSpectrum
from PySDM.products.state import SuperDropletCount
from PySDM.products.state import TotalParticleConcentration
from PySDM.products.state import TotalParticleSpecificConcentration
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
        rhod = make_rhod(self.setup.grid, self.setup.rhod)
        advector = nondivergent_vector_field_2d(
            self.setup.grid, self.setup.size, self.setup.dt, self.setup.stream_function)
        advectees = dict((key, np.full(self.setup.grid, value)) for key, value in self.setup.field_values.items())
        mpdatas = MPDATA(
            grid=self.setup.grid, dt=self.setup.dt,
            field_values=advectees,
            g_factor=rhod,
            advector=advector,
            mpdata_iters=self.setup.mpdata_iters,
            mpdata_infinite_gauge=self.setup.mpdata_iga,
            mpdata_flux_corrected_transport=self.setup.mpdata_fct,
            mpdata_third_order_terms=self.setup.mpdata_tot
        )
        builder.set_environment(MoistEulerian2DKinematic(
            dt=self.setup.dt,
            grid=self.setup.grid,
            size=self.setup.size,
            rhod_of=self.setup.rhod,
            eulerian_advection_solvers=mpdatas
        ))

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
        if self.setup.processes['fluid advection']:
            builder.add_dynamic(EulerianAdvection())
        if self.setup.processes["particle advection"]:
            displacement = Displacement(scheme='FTBS', enable_sedimentation=self.setup.processes["sedimentation"])
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
