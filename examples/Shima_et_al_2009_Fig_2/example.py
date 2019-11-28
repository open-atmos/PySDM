"""
Created at 08.08.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import copy
import numpy as np

from PySDM.simulation.particles import Particles
from PySDM.simulation.environment.box import Box
from PySDM.simulation.dynamics.coalescence.algorithms.sdm import SDM
from PySDM.simulation.initialisation.spectral_discretisation import constant_multiplicity

from examples.Shima_et_al_2009_Fig_2.setup import SetupA
from examples.Shima_et_al_2009_Fig_2.plotter import Plotter


def run(setup):
    particles = Particles(n_sd=setup.n_sd, dt=setup.dt, backend=setup.backend)
    particles.set_environment(Box, (setup.dv,))
    x, n = constant_multiplicity(setup.n_sd, setup.spectrum, (setup.x_min, setup.x_max))
    particles.create_state_0d(n=n, extensive={'x': x}, intensive={})
    particles.add_dynamics(SDM, (setup.kernel,))

    states = {}
    for step in setup.steps:
        particles.run(step - particles.n_steps)
        setup.check(particles.state, particles.n_steps)
        states[particles.n_steps] = copy.deepcopy(particles.state)

    return states, particles.stats


if __name__ == '__main__':
    with np.errstate(all='raise'):
        setup = SetupA()

        setup.n_sd = 2 ** 15
        setup.steps = [0, 90, 180]
        setup.check = lambda _, __: 0  # TODO!!!

        states, _ = run(setup)

        x_min = min([state.min('x') for state in states.values()])
        x_max = max([state.max('x') for state in states.values()])

    with np.errstate(invalid='ignore'):
        plotter = Plotter(setup, (x_min, x_max))
        for step, state in states.items():
            plotter.plot(state, step * setup.dt)
        plotter.show()

