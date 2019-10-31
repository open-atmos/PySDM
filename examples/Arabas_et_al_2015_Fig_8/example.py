"""
Created at 25.09.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np

from SDM.simulation.runner import Runner
from SDM.simulation.state import State
from SDM.simulation.dynamics import coalescence, advection
from SDM.simulation.discretisations import spatial, spectral
from SDM.simulation.maths import Maths

from examples.Arabas_et_al_2015_Fig_8.setup import Setup
from examples.Arabas_et_al_2015_Fig_8.storage import Storage
from examples.Arabas_et_al_2015_Fig_8.mpdata.mpdata_factory import MPDATAFactory


class DummyController:
    panic = False
    def set_percent(*_): pass
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

            dynamics = []
            # TODO: order of processes?
            if self.setup.processes["coalescence"]:
                dynamics.append(coalescence.SDM(self.setup.kernel, self.setup.dt, self.setup.dv, n_sd=self.setup.n_sd, n_cell=n_cell, backend=self.setup.backend))
            if self.setup.processes["advection"]:
                dynamics.append(advection.Advection(n_sd=self.setup.n_sd, courant_field=courant_field.data, scheme='FTBS', backend=self.setup.backend))

            runner = Runner(state, dynamics)
            moment_0 = np.empty(self.setup.grid)

            for step in self.setup.steps:
                if controller.panic:
                    break

                # async: Eulerian advection (TODO: run in background)
                #eulerian_fields.step() # TODO: same arg as run below!

                # async: coalescence and Lagrangian advection/sedimentation(TODO: run in the background)
                runner.run(step - runner.n_steps)

                # synchronous part:
                # - condensation

                # runner.state  # TODO: ...save()

                Maths.moment_2d(moment_0, state=state, k=0) 
                self.storage.save(moment_0 / self.setup.dv, step)

                controller.set_percent(float(step + 1) / self.setup.steps[-1])

        return runner.stats


def main():
    with np.errstate(all='raise'):
        setup = Setup()
        setup.check = lambda _, __: 0  # TODO!!!
        storage = Storage()
        simulation = Simulation(setup, storage)
        controller = DummyController()
        stats = simulation.run(controller)

    # with np.errstate(invalid='ignore'):
    #     plotter = Plotter(setup, (x_min, x_max))
    #     for step, state in states.items():
    #         plotter.plot(state, step * setup.dt)
    #     plotter.show()


if __name__ == '__main__':
    main()
