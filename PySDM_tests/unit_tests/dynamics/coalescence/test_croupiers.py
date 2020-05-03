"""
Created at 13.01.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import pytest
import numpy as np
from PySDM.initialisation.spectra import Lognormal
from PySDM.initialisation.spectral_sampling import linear
from PySDM_tests.unit_tests.state.testable_state_factory import TestableStateFactory
from PySDM_tests.unit_tests.state.dummy_particles import DummyParticles
from PySDM_tests.unit_tests.state.dummy_environment import DummyEnvironment
from PySDM.backends.default import Default
from PySDM.initialisation.multiplicities import discretise_n


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

    spectrum = Lognormal(n_part, v_mean, d)
    v, n = linear(n_sd, spectrum, (v_min, v_max))
    n = discretise_n(n)
    particles = DummyParticles(backend, n_sd)
    particles.set_environment(DummyEnvironment, {'grid': (x, y)})
    particles.croupier = croupier

    cell_id = backend.array((n_sd,), dtype=int)
    cell_origin_np = np.concatenate([np.random.randint(0, x, n_sd), np.random.randint(0, y, n_sd)]).reshape((-1, 2))
    cell_origin = backend.from_ndarray(cell_origin_np)
    position_in_cell_np = np.concatenate([np.random.rand(n_sd), np.random.rand(n_sd)]).reshape((-1, 2))
    position_in_cell = backend.from_ndarray(position_in_cell_np)
    state = TestableStateFactory.state(n=n, extensive={'volume': v}, intensive={}, cell_id=cell_id,
                                       cell_origin=cell_origin, position_in_cell=position_in_cell, particles=particles)
    particles.state = state

    # Act
    u01 = backend.from_ndarray(np.random.random(n_sd))
    particles.permute(u01)
    _ = particles.state.cell_start

    # Assert
    assert (np.diff(state.cell_id[state._State__idx]) >= 0).all()
