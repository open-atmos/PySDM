"""
Created at 25.09.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np

from SDM.simulation.runner import Runner
from SDM.simulation.state import State
from SDM.simulation.dynamics.coalescence import SDM
from SDM.simulation.discretisations import spatial, spectral

from examples.Arabas_et_al_2015_Fig_8.setup import Setup
from examples.Arabas_et_al_2015_Fig_8.mpdata.mpdata_factory import MPDATAFactory


# instantiation of simulation components, time-stepping
def run(setup):
    # Eulerian domain
    eulerian_fields = MPDATAFactory.kinematic_2d(grid=setup.grid, size=setup.size,
                                                 stream_function=setup.stream_function,
                                                 field_values=setup.field_values)

    # Lagrangian domain
    x, n = spectral.constant_multiplicity(setup.n_sd, setup.spectrum, (setup.x_min, setup.x_max))
    positions = spatial.pseudorandom(setup.grid, setup.n_sd)
    state = State.state_2d(n=n, extensive={'x': x}, intensive={}, positions=positions, backend=setup.backend)
    collider = SDM(setup.kernel, setup.dt, setup.dv, n_sd=setup.n_sd, backend=setup.backend)
    runner = Runner(state, (collider,))

    for step in setup.steps:
        # async: Eulerian advection (TODO: run in background)
        eulerian_fields.step()

        # async: coalescence and Lagrangian advection/sedimentation(TODO: run in the background)
        runner.run(step - runner.n_steps)

        # synchronous part:
        # - condensation

        # runner.state  # TODO: ...save()

    return runner.stats


if __name__ == '__main__':
    with np.errstate(all='raise'):
        setup = Setup()

        setup.check = lambda _, __: 0  # TODO!!!

        stats = run(setup)

        # x_min = min([state.min('x') for state in states.values()])
        # x_max = max([state.max('x') for state in states.values()])

    # with np.errstate(invalid='ignore'):
    #     plotter = Plotter(setup, (x_min, x_max))
    #     for step, state in states.items():
    #         plotter.plot(state, step * setup.dt)
    #     plotter.show()
