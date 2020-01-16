"""
Created at 08.08.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import copy

from PySDM.backends.default import Default
from PySDM.simulation.particles import Particles
from PySDM.simulation.dynamics.coalescence.algorithms.sdm import SDM
from PySDM.simulation.initialisation.spectral_sampling import constant_multiplicity
from PySDM.simulation.dynamics.coalescence.kernels.golovin import Golovin
from PySDM.simulation.initialisation.spectra import Exponential
from PySDM.simulation.environment.box import Box


backend = Default


def test_coalescence():
    # TODO: np.random.RandomState in backend?

    # Arrange
    v_min = 4.186e-15
    v_max = 4.186e-12
    n_sd = 2 ** 13
    steps = [0, 30, 60]
    X0 = 4 / 3 * 3.14 * 30.531e-6 ** 3
    n_part = 2 ** 23  # [m-3]
    dv = 1e6  # [m3]
    dt = 1  # [s]
    norm_factor = n_part * dv

    kernel = Golovin(b=1.5e3)  # [s-1]
    spectrum = Exponential(norm_factor=norm_factor, scale=X0)
    particles = Particles(n_sd=n_sd, dt=dt, backend=backend)
    particles.set_mesh_0d(dv=dv)
    particles.set_environment(Box, {})
    v, n = constant_multiplicity(n_sd, spectrum, (v_min, v_max))
    particles.create_state_0d(n=n, extensive={'volume': v}, intensive={})
    particles.add_dynamic(SDM, {"kernel": kernel})

    states = {}

    # Act
    for step in steps:
        particles.run(step - particles.n_steps)
        states[particles.n_steps] = copy.deepcopy(particles.state)

    # Assert
    x_max = 0
    for state in states.values():
        assert x_max < state.max('volume')
        x_max = state.max('volume')

