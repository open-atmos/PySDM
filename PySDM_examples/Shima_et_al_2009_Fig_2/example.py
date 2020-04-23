"""
Created at 08.08.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import copy
import numpy as np

from PySDM.simulation.particles_builder import ParticlesBuilder
from PySDM.simulation.environment.box import Box
from PySDM.simulation.dynamics.coalescence.algorithms.sdm import SDM
from PySDM.simulation.initialisation.spectral_sampling import constant_multiplicity

from PySDM_examples.Shima_et_al_2009_Fig_2.setup import SetupA
from PySDM_examples.Shima_et_al_2009_Fig_2.plotter import Plotter


def run(setup):
    particles_builder = ParticlesBuilder(n_sd=setup.n_sd, dt=setup.dt, backend=setup.backend)
    particles_builder.set_mesh_0d(setup.dv)
    particles_builder.set_environment(Box, {})
    v, n = constant_multiplicity(setup.n_sd, setup.spectrum, (setup.x_min, setup.x_max))
    particles_builder.create_state_0d(n=n, extensive={'volume': v}, intensive={})
    particles_builder.register_dynamic(SDM, {"kernel": setup.kernel})
    particles = particles_builder.get_particles()

    states = {}
    for step in setup.steps:
        particles.run(step - particles.n_steps)
        setup.check(particles.state, particles.n_steps)
        states[particles.n_steps] = copy.deepcopy(particles.state)

    return states, particles.stats


def main(plot:bool):
    with np.errstate(all='raise'):
        setup = SetupA()

        setup.n_sd = 2 ** 15
        setup.check = lambda _, __: 0  # TODO!!!

        states, _ = run(setup)

        x_min = min([state.min('volume') for state in states.values()])
        x_max = max([state.max('volume') for state in states.values()])

    with np.errstate(invalid='ignore'):
        plotter = Plotter(setup, (x_min, x_max))
        for step, state in states.items():
            plotter.plot(state, step * setup.dt)
        if plot:
            plotter.show()


if __name__ == '__main__':
    main(plot=True)
