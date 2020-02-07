from PySDM_tests.unit_tests.simulation.state.testable_state_factory import TestableStateFactory
from PySDM_tests.unit_tests.simulation.state.dummy_particles import DummyParticles
from PySDM_tests.unit_tests.simulation.dynamics.advection.dummy_environment import DummyEnvironment
from PySDM.backends.default import Default

import numpy as np
import pytest

backend = Default


class TestState:

    @staticmethod
    def storage(iterable):
        return backend.from_ndarray(np.array(iterable))

    @pytest.mark.parametrize("volume, n", [
        pytest.param(np.array([1., 1, 1, 1]), np.array([1, 1, 1, 1])),
        pytest.param(np.array([1., 2, 1, 1]), np.array([2, 0, 2, 0])),
        pytest.param(np.array([1., 1, 4]), np.array([5, 0, 0]))
    ])
    def test_housekeeping(self, volume, n):
        # Arrange
        particles = DummyParticles(backend, n_sd=len(n))
        sut = TestableStateFactory.state_0d(n=n, extensive={'volume': volume}, intensive={}, particles=particles)
        # TODO
        sut.healthy = particles.backend.from_ndarray(np.array([0]))

        # Act
        sut.housekeeping()

        # Assert
        assert sut['volume'].shape == sut['n'].shape
        assert sut.SD_num == (n != 0).sum()
        assert sut['n'].sum() == n.sum()
        assert (sut['volume'] * sut['n']).sum() == (volume * n).sum()

    def test_sort_by_cell_id(self):
        # Arrange
        particles = DummyParticles(backend, n_sd=3)
        sut = TestableStateFactory.empty_state(particles)
        sut.n = TestState.storage([0, 1, 0, 1, 1])
        cells = [3, 4, 0, 1, 2]
        n_cell = max(cells) + 1
        sut.cell_id = TestState.storage(cells)
        sut._State__idx = TestState.storage([4, 1, 3, 2, 0])
        sut._State__tmp_idx = TestState.storage([0] * 5)
        sut._State__cell_start = TestState.storage([0] * (n_cell + 1))
        sut._State__cell_start_p = backend.array((2, len(sut._State__cell_start)), dtype=int)
        sut.SD_num = particles.n_sd

        # Act
        sut._State__sort_by_cell_id()

        # Assert
        assert len(sut._State__idx) == 5
        np.testing.assert_array_equal(np.array([3, 4, 1]), backend.to_ndarray(sut._State__idx[:sut.SD_num]))

    def test_recalculate_cell_id(self):
        # Arrange
        n = np.ones(1)
        droplet_id = 0
        initial_position = Default.from_ndarray(np.array([[0, 0]]))
        grid = (1, 1)
        particles = DummyParticles(backend, n_sd=1)
        particles.set_mesh(grid)
        particles.set_environment(DummyEnvironment, (None,))
        sut = TestableStateFactory.state_2d(n=n, intensive={}, extensive={},
                                            particles=particles, positions=initial_position)
        sut.cell_origin[droplet_id, 0] = .1
        sut.cell_origin[droplet_id, 1] = .2
        sut.cell_id[droplet_id] = -1

        # Act
        sut.recalculate_cell_id()

        # Assert
        assert sut.cell_id[droplet_id] == 0


