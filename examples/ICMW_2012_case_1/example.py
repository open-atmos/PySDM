"""
Created at 25.09.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import numpy as np

from PySDM.simulation.particles import Particles as Particles
from PySDM.simulation.dynamics.advection import Advection
from PySDM.simulation.dynamics.condensation import Condensation
from PySDM.simulation.dynamics.coalescence.algorithms.sdm import SDM
from PySDM.simulation.initialisation import spatial_discretisation, spectral_discretisation
from PySDM.simulation.environment.moist_air import MoistAir
from PySDM import utils

from examples.ICMW_2012_case_1.setup import Setup
from examples.ICMW_2012_case_1.storage import Storage
from MPyDATA.mpdata.mpdata_factory import MPDATAFactory


class DummyController:
    panic = False

    def set_percent(self, value): print(f"{100 * value:.1f}%")

    def __enter__(*_): pass

    def __exit__(*_): pass


class Simulation:
    def __init__(self, setup, storage):
        self.setup = setup
        self.storage = storage
        self.tmp = None
        self.particles = Particles(n_sd=self.setup.n_sd,
                                   dt=self.setup.dt,
                                   size=self.setup.size,
                                   grid=self.setup.grid,
                                   backend=self.setup.backend)

    # instantiation of simulation components, time-stepping
    def run(self, controller=DummyController()):
        courant_field, eulerian_fields = MPDATAFactory.kinematic_2d(
            grid=self.setup.grid, size=self.setup.size, dt=self.setup.dt,
            stream_function=self.setup.stream_function,
            field_values=self.setup.field_values)

        self.particles.set_environment(MoistAir, (
            lambda: eulerian_fields.mpdatas["th"].curr.get(),
            lambda: eulerian_fields.mpdatas["qv"].curr.get(),
            self.setup.rhod
        ))

        self.particles.create_state_2d2( # TODO: ...
                                        extensive={},
                                        intensive={},
                                        spatial_discretisation=spatial_discretisation.pseudorandom,
                                        spectral_discretisation=spectral_discretisation.constant_multiplicity,
                                        spectrum_per_mass_of_dry_air=self.setup.spectrum_per_mass_of_dry_air,
                                        r_range=(self.setup.r_min, self.setup.r_max),
                                        kappa=self.setup.kappa
        )

        if self.setup.processes["coalescence"]:
            self.particles.add_dynamics(SDM, (self.setup.kernel,))
        if self.setup.processes["advection"]:
            courant_field_data = [courant_field.data(0), courant_field.data(1)]
            self.particles.add_dynamics(Advection, (courant_field_data, 'FTBS'))
        if self.setup.processes["condensation"]:
            self.particles.add_dynamics(Condensation, (self.particles.environment, self.setup.kappa))

        # TODO
        if self.storage is not None:
            self.storage.init(self.setup)

        with controller:
            for step in self.setup.steps:
                if controller.panic:
                    break

                for _ in range(step - self.particles.n_steps):
                    if self.setup.processes["advection"]:
                        eulerian_fields.step()

                    self.particles.run(1)

                self.store(self.particles.state, eulerian_fields, self.particles.environment, step)

                controller.set_percent(step / self.setup.steps[-1])

        return self.particles.stats

    def store(self, state, eulerian_fields, ambient_air, step):
        # allocations
        if self.tmp is None:  # TODO: move to constructor
            n_moments = 0
            for attr in self.setup.specs:
                for _ in self.setup.specs[attr]:
                    n_moments += 1
            self.moment_0 = state.backend.array(state.n_cell, dtype=int)
            self.moments = state.backend.array((n_moments, state.n_cell), dtype=float)
            self.tmp = np.empty(state.n_cell)

        # store moments
        state.moments(self.moment_0, self.moments, self.setup.specs)  # TODO: attr_range
        state.backend.download(self.moment_0, self.tmp)
        self.tmp /= self.setup.dv
        self.storage.save(self.tmp.reshape(self.setup.grid), step, "m0")

        i = 0
        for attr in self.setup.specs:
            for k in self.setup.specs[attr]:
                state.backend.download(self.moments[i], self.tmp)  # TODO: [i] will not work
                self.tmp /= self.setup.dv
                self.storage.save(self.tmp.reshape(self.setup.grid), step, f"{attr}_m{k}")
                i += 1

        # store advected fields
        for key in eulerian_fields.mpdatas.keys():
            self.storage.save(eulerian_fields.mpdatas[key].curr.get(), step, key)

        # store auxiliary fields
        state.backend.download(ambient_air.RH, self.tmp)
        self.storage.save(self.tmp.reshape(self.setup.grid), step, "RH")


def main():
    # with np.errstate(all='raise'):
    setup = Setup()
    storage = Storage()
    simulation = Simulation(setup, storage)
    simulation.run()


if __name__ == '__main__':
    main()
