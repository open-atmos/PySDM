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
from examples.Arabas_et_al_2015_Fig_8.mpdata.mpdata_factory import MPDATAFactory


# instantiation of simulation components, time-stepping
def run(setup):
    # Eulerian domain
    courant_field, eulerian_fields = MPDATAFactory.kinematic_2d(grid=setup.grid, size=setup.size, dt=setup.dt,
                                                 stream_function=setup.stream_function,
                                                 field_values=setup.field_values)

    # Lagrangian domain
    x, n = spectral.constant_multiplicity(setup.n_sd, setup.spectrum, (setup.x_min, setup.x_max))
    n[0] *= 20
    positions = spatial.pseudorandom(setup.grid, setup.n_sd)
    state = State.state_2d(n=n, grid=setup.grid, extensive={'x': x}, intensive={}, positions=positions,
                           backend=setup.backend)
    n_cell = setup.grid[0] * setup.grid[1]

    dynamics = (
        #coalescence.SDM(setup.kernel, setup.dt, setup.dv, n_sd=setup.n_sd, n_cell=n_cell, backend=setup.backend),
        advection.ExplicitEulerWithInterpolation(n_sd=setup.n_sd, courant_field=courant_field.data, backend=setup.backend),
    )
    runner = Runner(state, dynamics)
    moment_0 = np.empty(setup.grid)

    for step in setup.steps:
        # async: Eulerian advection (TODO: run in background)
        #eulerian_fields.step() # TODO: same arg as run below!

        # async: coalescence and Lagrangian advection/sedimentation(TODO: run in the background)
        runner.run(step - runner.n_steps)

        # synchronous part:
        # - condensation

        # runner.state  # TODO: ...save()

        Maths.moment_2d(moment_0, state=state, k=0)
        import matplotlib.pyplot as plt
        plt.imshow(moment_0)
        plt.show()

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
