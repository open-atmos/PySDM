"""
Created at 25.09.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import numpy as np

from PySDM.simulation.runner import Runner
from PySDM.simulation.state import State
from PySDM.simulation.dynamics import coalescence, advection, condensation
from PySDM.simulation.discretisations import spatial, spectral
from PySDM.simulation.maths import Maths

from examples.Arabas_et_al_2015_Fig_8.setup import Setup
from examples.Arabas_et_al_2015_Fig_8.storage import Storage
from MPyDATA.mpdata.mpdata_factory import MPDATAFactory


class DummyController:
    panic = False
    def set_percent(self, value): print(f"{100*value:.1f}%")
    def __enter__(*_): pass
    def __exit__(*_): pass


class Simulation:
    def __init__(self, setup, storage):
        self.setup = setup
        self.storage = storage

    # instantiation of simulation components, time-stepping
    def run(self, controller):
        self.storage.init(self.setup)
        with controller:
            # Eulerian domain
            courant_field, eulerian_fields = MPDATAFactory.kinematic_2d(
                grid=self.setup.grid, size=self.setup.size, dt=self.setup.dt,
                stream_function=self.setup.stream_function,
                field_values=self.setup.field_values)

            # Lagrangian domain
            x, n = spectral.constant_multiplicity(self.setup.n_sd, self.setup.spectrum, (self.setup.x_min, self.setup.x_max))
            n[0] *= 20
            positions = spatial.pseudorandom(self.setup.grid, self.setup.n_sd)
            state = State.state_2d(n=n, grid=self.setup.grid, extensive={'x': x}, intensive={}, positions=positions,
				   backend=self.setup.backend)
            n_cell = self.setup.grid[0] * self.setup.grid[1]

            ambient_air = self.setup.ambient_air(
                grid=self.setup.grid,
                backend=self.setup.backend,
                thd_lambda=lambda: eulerian_fields.mpdatas["th"].curr.get(),
                qv_lambda=lambda: eulerian_fields.mpdatas["qv"].curr.get()
            )

            dynamics = []
            # TODO: order of processes?
            if self.setup.processes["coalescence"]:
                dynamics.append(coalescence.SDM(self.setup.kernel, self.setup.dt, self.setup.dv, n_sd=self.setup.n_sd, n_cell=n_cell, backend=self.setup.backend))
            if self.setup.processes["advection"]:
                dynamics.append(advection.Advection(n_sd=self.setup.n_sd, courant_field=courant_field.data, scheme='FTBS', backend=self.setup.backend))
            if self.setup.processes["condensation"]:
                dynamics.append(condensation.Condensation(ambient_air))

            runner = Runner(state, dynamics)

            for step in self.setup.steps:
                if controller.panic:
                    break

                for _ in range(step - runner.n_steps):
                    # async: Eulerian advection (TODO: run in background)
                    if self.setup.processes["advection"]:
                        eulerian_fields.step()

                    # async: coalescence and Lagrangian advection/sedimentation(TODO: run in the background)
                    runner.run(1)

                # synchronous part:
                # - condensation:
                #   - TODO: update fields due to condensation/evaporation
                #   - TODO: ensure the above does include droplets that precipitated out of the domain

                self.store(state, eulerian_fields, ambient_air, step)

                controller.set_percent(step / self.setup.steps[-1])

        return runner.stats

    def store(self, state, eulerian_fields, ambient_air, step):
        # store moments
        moment_0 = np.empty(self.setup.grid)
        Maths.moment_2d(moment_0, state=state, k=0)
        self.storage.save(moment_0 / self.setup.dv, step, "m0")

        # store advected fields
        for key in eulerian_fields.mpdatas.keys():
            self.storage.save(eulerian_fields.mpdatas[key].curr.get(), step, key)

        # store auxiliary fields (TODO: assumes numpy backend)
        self.storage.save(ambient_air.RH.reshape(self.setup.grid), step, "RH")


def main():
    with np.errstate(all='raise'):
        setup = Setup()
        setup.check = lambda _, __: 0  # TODO!!!
        storage = Storage()
        simulation = Simulation(setup, storage)
        controller = DummyController()
        simulation.run(controller)


if __name__ == '__main__':
    main()
