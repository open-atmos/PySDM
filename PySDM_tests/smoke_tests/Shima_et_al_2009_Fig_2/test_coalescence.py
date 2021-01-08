"""
Created at 08.08.2019
"""

import copy

import numpy as np
import pytest

from PySDM.backends import CPU
from PySDM.builder import Builder
from PySDM.dynamics import Coalescence
from PySDM.dynamics.coalescence.kernels import Golovin
from PySDM.environments import Box
from PySDM.initialisation.spectra import Exponential
from PySDM.initialisation.spectral_sampling import ConstantMultiplicity
from PySDM.physics.constants import si
from PySDM_tests.backends_fixture import backend


def check(n_part, dv, n_sd, rho, state, step):
    check_LWC = 1e-3 * si.kilogram / si.metre ** 3
    check_ksi = n_part * dv / n_sd

    # multiplicities
    if step == 0:
        np.testing.assert_approx_equal(np.amin(state['n']), np.amax(state['n']), 1)
        np.testing.assert_approx_equal(state['n'][0], check_ksi, 1)

    # liquid water content
    LWC = rho * np.dot(state['n'], state['volume']) / dv
    np.testing.assert_approx_equal(LWC, check_LWC, 3)


@pytest.mark.parametrize('croupier', ['local', 'global'])
def test_coalescence(backend, croupier):
    # Arrange
    n_sd = 2 ** 13
    steps = [0, 30, 60]
    X0 = 4 / 3 * np.pi * 30.531e-6 ** 3
    n_part = 2 ** 23 / si.metre ** 3
    dv = 1e6 * si.metres ** 3
    dt = 1 * si.seconds
    norm_factor = n_part * dv
    rho = 1000 * si.kilogram / si.metre ** 3

    kernel = Golovin(b=1.5e3)  # [s-1]
    spectrum = Exponential(norm_factor=norm_factor, scale=X0)
    builder = Builder(n_sd=n_sd, backend=backend)
    builder.set_environment(Box(dt=dt, dv=dv))
    attributes = {}
    attributes['volume'], attributes['n'] = ConstantMultiplicity(spectrum).sample(n_sd)
    builder.add_dynamic(Coalescence(kernel, seed=256))
    core = builder.build(attributes)
    core.croupier = croupier

    volumes = {}

    # Act
    for step in steps:
        core.run(step - core.n_steps)
        check(n_part, dv, n_sd, rho, core.particles, step)
        volumes[core.n_steps] = core.particles['volume'].to_ndarray()

    # Assert
    x_max = 0
    for volume in volumes.values():
        assert x_max < np.amax(volume)
        x_max = np.amax(volume)

