"""
Created at 08.08.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np

from PySDM.simulation.particles_builder import ParticlesBuilder
from PySDM.simulation.environment.box import Box
from PySDM.simulation.dynamics.coalescence.algorithms.sdm import SDM
from PySDM.simulation.initialisation.spectral_sampling import constant_multiplicity

from PySDM_examples.Shima_et_al_2009_Fig_2.setup import SetupA
from PySDM_examples.Shima_et_al_2009_Fig_2.plotter import Plotter


def run(setup):
    particles_builder = ParticlesBuilder(n_sd=setup.n_sd, dt=setup.dt, backend=setup.backend)
    particles_builder.set_environment(Box, {"dv": setup.dv})
    v, n = constant_multiplicity(setup.n_sd, setup.spectrum, (setup.init_x_min, setup.init_x_max))
    particles_builder.create_state_0d(n=n, extensive={'volume': v}, intensive={})
    particles_builder.register_dynamic(SDM, {"kernel": setup.kernel})
    particles = particles_builder.get_particles()

    class Seed:
        seed = 0

        def __call__(self):
            Seed.seed += 1
            return Seed.seed
    particles.dynamics[str(SDM)].seed = Seed()

    vals = {}
    for step in setup.steps:
        particles.run(step - particles.n_steps)
        vals[step] = particles.products['dv/dlnr'].get(setup.radius_bins_edges)
        vals[step][:] *= setup.rho

    return vals, particles.stats


def main(plot: bool):
    with np.errstate(all='raise'):
        setup = SetupA()

        setup.n_sd = 2 ** 15

        states, _ = run(setup)

    with np.errstate(invalid='ignore'):
        plotter = Plotter(setup)
        for step, vals in states.items():
            plotter.plot(vals, step * setup.dt)
        if plot:
            plotter.show()


if __name__ == '__main__':
    main(plot=True)
