"""
Created at 08.08.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import copy
import numpy as np

from SDM.simulation.runner import Runner
from SDM.simulation.state import State
from SDM.simulation.colliders import SDM
from SDM.simulation.discretisations import constant_multiplicity
from examples.Shima_et_al_2009_Fig_2.setup import SetupA
from examples.Shima_et_al_2009_Fig_2.plotter import Plotter

#%%


def run(setup):
    x, n = constant_multiplicity(setup.n_sd, setup.spectrum, (setup.x_min, setup.x_max))
    state = State(n=n, extensive={'x': x}, intensive={}, segment_num=1, backend=setup.backend)
    collider = SDM(setup.kernel, setup.dt, setup.dv, n_sd=setup.n_sd, backend=setup.backend)
    runner = Runner(state, (collider,))

    states = {}
    for step in setup.steps:
        runner.run(step - runner.n_steps)
        # setup.check(runner.state, runner.n_steps) TODO!!!
        states[runner.n_steps] = copy.deepcopy(runner.state)

    return states, runner.stats

#%%


with np.errstate(all='raise'):
    setup = SetupA()

    setup.backend.init()

    setup.n_sd = 2 ** 18
    # setup.steps = [0, 1]

    states, _ = run(setup)

    x_min = min([state.min('x') for state in states.values()])
    x_max = max([state.max('x') for state in states.values()])

with np.errstate(invalid='ignore'):
    plotter = Plotter(setup, (x_min, x_max))
    for step, state in states.items():
        plotter.plot(state, step * setup.dt)
    plotter.show()

#%%
