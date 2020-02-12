"""
Created at 25.09.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import time

from PySDM.simulation.particles import Particles as Particles
from PySDM.simulation.dynamics.advection import Advection
from PySDM.simulation.dynamics.condensation.condensation import Condensation
from PySDM.simulation.dynamics.eulerian_advection import EulerianAdvection
from PySDM.simulation.dynamics.coalescence.algorithms.sdm import SDM
from PySDM.simulation.initialisation import spatial_sampling, spectral_sampling
from PySDM.simulation.environment.moist_eulerian_2d_kinematic import MoistEulerian2DKinematic


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

    @property
    def products(self):
        return self.particles.products

    def reinit(self):
        self.tmp = None  # TODO!
        self.particles = Particles(n_sd=self.setup.n_sd, dt=self.setup.dt, backend=self.setup.backend)
        self.particles.set_mesh(grid=self.setup.grid, size=self.setup.size)
        self.particles.set_environment(MoistEulerian2DKinematic, {
            "stream_function": self.setup.stream_function,
            "field_values": self.setup.field_values,
            "rhod_of": self.setup.rhod,
            "mpdata_iga": self.setup.mpdata_iga,
            "mpdata_tot": self.setup.mpdata_tot,
            "mpdata_fct": self.setup.mpdata_fct,
            "mpdata_iters": self.setup.mpdata_iters
        })

        self.particles.create_state_2d(
            extensive={},
            intensive={},
            spatial_discretisation=spatial_sampling.pseudorandom,
            spectral_discretisation=spectral_sampling.constant_multiplicity,  # TODO: random
            spectrum_per_mass_of_dry_air=self.setup.spectrum_per_mass_of_dry_air,
            r_range=(self.setup.r_min, self.setup.r_max),
            kappa=self.setup.kappa,
            radius_threshold = self.setup.aerosol_radius_threshold
        )

        if self.setup.processes["condensation"]:
            self.particles.register_dynamic(Condensation, {
                "kappa": self.setup.kappa,
                "scheme": self.setup.condensation_scheme,
                "rtol_lnv": self.setup.condensation_rtol_lnv,
                "rtol_thd": self.setup.condensation_rtol_thd,
            })
            self.particles.register_dynamic(EulerianAdvection, {})
        if self.setup.processes["advection"]:
            self.particles.register_dynamic(Advection, {"scheme": 'FTBS', "sedimentation": self.setup.processes["sedimentation"]})
        if self.setup.processes["coalescence"]:
            self.particles.register_dynamic(SDM, {"kernel": self.setup.kernel})

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
