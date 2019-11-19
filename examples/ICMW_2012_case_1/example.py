"""
Created at 25.09.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""


import numpy as np

from PySDM.simulation.runner import Runner
from PySDM.simulation.state.state_factory import StateFactory
from PySDM.simulation.dynamics import advection, condensation
from PySDM.simulation.dynamics.coalescence.algorithms import sdm
from PySDM.simulation.environment.moist_air import MoistAir
from PySDM.simulation.initialisation import spatial_discretisation, spectral_discretisation
from PySDM.simulation.initialisation.r_wet_init import r_wet_init
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

    def init(self):
        self.tmp = None # TODO (done to force reinitialisation of tmp arrays as setup.grid might have changed)

        # TODO: particles = Particles(self.setup.backend)
        courant_field, self.eulerian_fields = MPDATAFactory.kinematic_2d(
            grid=self.setup.grid, size=self.setup.size, dt=self.setup.dt,
            stream_function=self.setup.stream_function,
            field_values=self.setup.field_values)

        self.ambient_air = MoistAir(
            grid=self.setup.grid,
            backend=self.setup.backend,
            thd_xzt_lambda=lambda: self.eulerian_fields.mpdatas["th"].curr.get(),
            qv_xzt_lambda=lambda: self.eulerian_fields.mpdatas["qv"].curr.get(),
            rhod_z_lambda=self.setup.rhod
        )

        positions = spatial_discretisation.pseudorandom(self.setup.grid, self.setup.n_sd)

        # <TEMP>
        cell_origin = positions.astype(dtype=int)
        strides = utils.strides(self.setup.grid)
        cell_id = np.dot(strides, cell_origin.T).ravel()
        # </TEMP>

        with np.errstate(all='raise'):
            # number per unit mass of dry air
            r_dry, n = spectral_discretisation.constant_multiplicity(
                self.setup.n_sd, self.setup.spectrum_per_mass_of_dry_air, (self.setup.r_min, self.setup.r_max)
            )
            # number per unit volume of dry air
            for i in range(self.setup.n_sd):
                n[i] *= self.ambient_air.rhod[cell_id[i]]
            # number (in the whole domain)
            for i in range(self.setup.n_sd):
                n[i] *= self.setup.size[0] * self.setup.size[1]

        r_wet = r_wet_init(r_dry, self.ambient_air, cell_id, self.setup.kappa)
        self.state = StateFactory.state_2d(n=n, grid=self.setup.grid,
                                      # TODO: rename x -> ...
                                      extensive={
                                          'x': utils.Physics.r2x(r_wet),
                                          'dry volume': utils.Physics.r2x(r_dry)
                                      },
                                      intensive={},
                                      positions=positions,
                                      backend=self.setup.backend)
        n_cell = self.setup.grid[0] * self.setup.grid[1]

        self.dynamics = []
        if self.setup.processes["coalescence"]:
            self.dynamics.append(sdm.SDM(self.setup.kernel, self.setup.dt, self.setup.dv, n_sd=self.setup.n_sd,
                                    n_cell=n_cell, backend=self.setup.backend))
        if self.setup.processes["advection"]:
            courant_field_data = [courant_field.data(0), courant_field.data(1)]
            self.dynamics.append(advection.Advection(n_sd=self.setup.n_sd, courant_field=courant_field_data,
                                                scheme='FTBS', backend=self.setup.backend))
        if self.setup.processes["condensation"]:
            self.dynamics.append(condensation.Condensation(self.ambient_air, self.setup.dt, self.setup.kappa, self.setup.backend, n_cell))

    def run(self, controller):
        self.storage.init(self.setup)
        runner = Runner(self.state, self.dynamics)
        with controller:
            for step in self.setup.steps:
                if controller.panic:
                    break

                for _ in range(step - runner.n_steps):
                    if self.setup.processes["advection"]:
                        self.eulerian_fields.step()

                    runner.run(1)

                self.store(step)

                controller.set_percent(step / self.setup.steps[-1])

        return runner.stats

    def store(self, step):
        # allocations
        if self.tmp is None:  # TODO: move to constructor
            n_moments = 0
            for attr in self.setup.specs:
                for _ in self.setup.specs[attr]:
                    n_moments += 1
            self.moment_0 = self.state.backend.array(self.state.n_cell, dtype=int)
            self.moments = self.state.backend.array((n_moments, self.state.n_cell), dtype=float)
            self.tmp = np.empty(self.state.n_cell)

        # store moments
        self.state.moments(self.moment_0, self.moments, self.setup.specs)  # TODO: attr_range
        self.state.backend.download(self.moment_0, self.tmp)
        self.tmp /= self.setup.dv
        self.storage.save(self.tmp.reshape(self.setup.grid), step, "m0")

        i = 0
        for attr in self.setup.specs:
            for k in self.setup.specs[attr]:
                self.state.backend.download(self.moments[i], self.tmp)  # TODO: [i] will not work
                self.tmp /= self.setup.dv
                self.storage.save(self.tmp.reshape(self.setup.grid), step, f"{attr}_m{k}")
                i += 1

        # store advected fields
        for key in self.eulerian_fields.mpdatas.keys():
            self.storage.save(self.eulerian_fields.mpdatas[key].curr.get(), step, key)

        # store auxiliary fields
        self.state.backend.download(self.ambient_air.RH, self.tmp)
        self.storage.save(self.tmp.reshape(self.setup.grid), step, "RH")


def main():
    #with np.errstate(all='raise'):
    setup = Setup()
    storage = Storage()
    simulation = Simulation(setup, storage)
    controller = DummyController()

    simulation.init()
    simulation.run(controller)


if __name__ == '__main__':
    main()
