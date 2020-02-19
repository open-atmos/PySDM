"""
Created at 08.08.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
import copy
import pytest
from PySDM.backends.default import Default
from PySDM.simulation.particles_builder import ParticlesBuilder
from PySDM.simulation.dynamics.coalescence.algorithms.sdm import SDM
from PySDM.simulation.initialisation.spectral_sampling import constant_multiplicity
from PySDM.simulation.dynamics.coalescence.kernels.golovin import Golovin
from PySDM.simulation.initialisation.spectra import Exponential
from PySDM.simulation.environment.box import Box


backend = Default


@pytest.mark.skip
@pytest.mark.parametrize('croupier', ['local', 'global'])
def test_coalescence(croupier):
    np.random.seed(0)  # TODO: working only for backend based on numpy

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
    particles_builder = ParticlesBuilder(n_sd=n_sd, dt=dt, backend=backend)
    particles_builder.set_mesh_0d(dv=dv)
    particles_builder.set_environment(Box, {})
    v, n = constant_multiplicity(n_sd, spectrum, (v_min, v_max))
    particles_builder.create_state_0d(n=n, extensive={'volume': v}, intensive={})
    particles_builder.register_dynamic(SDM, {"kernel": kernel})
    particles = particles_builder.get_particles()
    particles.croupier = croupier

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

