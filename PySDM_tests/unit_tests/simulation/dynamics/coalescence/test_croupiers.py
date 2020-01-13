"""
Created at 13.01.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import pytest
import numpy as np
from PySDM.simulation.dynamics.coalescence.croupiers import local_FisherYates, global_FisherYates
from PySDM.simulation.initialisation.spectra import Lognormal
from PySDM.simulation.initialisation.spectral_sampling import linear
from PySDM_tests.unit_tests.simulation.state.testable_state_factory import TestableStateFactory
from PySDM.simulation.particles import discretise_n
from PySDM_tests.unit_tests.simulation.state.dummy_particles import DummyParticles
from PySDM.backends.default import Default

backend = Default


@pytest.mark.parametrize('croupier', [local_FisherYates, global_FisherYates])
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
    particles.set_mesh((x, y))

    cell_id = backend.array((n_sd,), dtype=int)
    cell_origin_np = np.concatenate([np.random.randint(0, x, n_sd), np.random.randint(0, y, n_sd)]).reshape((-1, 2))
    cell_origin = backend.from_ndarray(cell_origin_np)
    position_in_cell_np = np.concatenate([np.random.rand(n_sd), np.random.rand(n_sd)]).reshape((-1, 2))
    position_in_cell = backend.from_ndarray(position_in_cell_np)
    state = TestableStateFactory.state(n=n, extensive={'volume': v}, intensive={}, cell_id=cell_id,
                                       cell_origin=cell_origin, position_in_cell=position_in_cell, particles=particles)
    cell_start = backend.array((x*y+1,), dtype=int)
    u01 = backend.from_ndarray(np.random.random(n_sd))

    # Act
    croupier(state, cell_start, u01)

    # Assert
    assert (np.diff(state.cell_id[state.idx]) >= 0).all()
