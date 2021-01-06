"""
Created at 23.12.2020
"""

import numpy as np
import pytest

from PySDM.environments import Box
from PySDM.initialisation.spatial_sampling import Pseudorandom
from PySDM.state.mesh import Mesh
# noinspection PyUnresolvedReferences
from PySDM_tests.backends_fixture import backend
from PySDM_tests.unit_tests.dynamics.coalescence.__parametrisation__ import get_dummy_core_and_sdm


class TestSDMSingleCell:

    @staticmethod
    @pytest.mark.parametrize("adaptive", [False, True])  # TODO: + False
    def test_coalescence_call(backend, adaptive):
        # Arrange
        n = np.ones(8000)
        v = np.ones_like(n)
        env = Box(dv=1, dt=0)
        grid = (25, 25)
        env.mesh = Mesh(grid, size=grid)
        core, sut = get_dummy_core_and_sdm(backend, len(n), environment=env)
        cell_id, _, _ = env.mesh.cellular_attributes(Pseudorandom.sample(grid, len(n)))
        attributes = {'n': n, 'volume': v, 'cell id': cell_id}
        core.build(attributes)
        u01, _ = sut.rnd_opt.get_random_arrays(s=0)
        sut.actual_length = core.particles._Particles__idx.length
        sut.adaptive = adaptive

        # Act
        sut()

        # Assert
        np.testing.assert_array_equal(cell_id, core.particles['cell id'].to_ndarray(raw=True))
