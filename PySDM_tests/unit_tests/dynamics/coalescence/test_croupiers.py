"""
Created at 13.01.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
import pytest

from PySDM.backends.default import Default
from PySDM.initialisation.spectra import Lognormal
from PySDM.initialisation.spectral_sampling import linear
from PySDM_tests.unit_tests.state.dummy_environment import DummyEnvironment
from PySDM_tests.unit_tests.state.dummy_particles import DummyParticles

backend = Default


@pytest.mark.parametrize('croupier', ['local', 'global'])
def test_final_state(croupier):
    # Arrange
    n_part = 10000
    v_mean = 2e-6
    d = 1.2
    v_min = 0.01e-6
    v_max = 10e-6
    n_sd = 64
    x = 4
    y = 4

    attributes = {}
    spectrum = Lognormal(n_part, v_mean, d)
    attributes['volume'], attributes['n'] = linear(n_sd, spectrum, (v_min, v_max))
    particles = DummyParticles(backend, n_sd)
    particles.set_environment(DummyEnvironment, {'grid': (x, y)})
    particles.croupier = croupier

    attributes['cell id'] = backend.array((n_sd,), dtype=int)
    cell_origin_np = np.concatenate([np.random.randint(0, x, n_sd), np.random.randint(0, y, n_sd)]).reshape((2, -1))
    attributes['cell origin'] = backend.from_ndarray(cell_origin_np)
    position_in_cell_np = np.concatenate([np.random.rand(n_sd), np.random.rand(n_sd)]).reshape((2, -1))
    attributes['position in cell'] = backend.from_ndarray(position_in_cell_np)
    particles.get_particles(attributes)

    # Act
    u01 = backend.Storage.from_ndarray(np.random.random(n_sd))
    particles.permute(u01)
    _ = particles.state.cell_start

    # Assert
    assert (np.diff(particles.state['cell id'][particles.state._State__idx]) >= 0).all()
