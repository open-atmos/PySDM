"""
Created at 07.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import copy
import numpy as np

from examples.Shima_et_al_2009_Fig2.plotter import Plotter
from examples.Shima_et_al_2009_Fig2.setups import SetupA

from SDM.runner import Runner
from SDM.state import State
from SDM.colliders import SDM
from SDM.undertakers import Resize
from SDM.discretisations import constant_multiplicity


def test_Fig2():
    with np.errstate(all='raise'):
        setup = SetupA()
        states, _ = run(setup)

        x_min = min([state.min('x') for state in states.values()])
        x_max = max([state.max('x') for state in states.values()])

    with np.errstate(invalid='ignore'):
        plotter = Plotter(setup, (x_min, x_max))
        for step, state in states.items():
            plotter.plot(state, step * setup.dt)
        plotter.show()


# TODO python -O
def test_timing():
    setup = SetupA()
    setup.steps = [100, 3600]

    nsds = [2 ** n for n in range(12, 15)]
    times = []
    for sd in nsds:
        setup.n_sd = sd
        _, stats = run(setup)
        times.append(stats.times[-1])

    from matplotlib import pyplot as plt
    plt.plot(nsds, times)
    plt.show()


def run(setup):
    x, n = constant_multiplicity(setup.n_sd, setup.spectrum, (setup.x_min, setup.x_max))
    state = State({'x': x, 'n': n})
    collider = SDM(setup.kernel, setup.dt, setup.dv)
    undertaker = Resize()
    runner = Runner(state, (undertaker, collider))

    states = {}
    for step in setup.steps:
        runner.run(step - runner.n_steps)
        setup.check(runner.state, runner.n_steps)
        states[runner.n_steps] = copy.deepcopy(runner.state)

    return states, runner.stats
