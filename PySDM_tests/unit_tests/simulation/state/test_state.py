from PySDM_tests.unit_tests.simulation.state.testable_state_factory import TestableStateFactory
from PySDM_tests.unit_tests.simulation.state.dummy_particles import DummyParticles
from PySDM_tests.unit_tests.simulation.state.dummy_environment import DummyEnvironment
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
        sut.healthy = TestState.storage([0])

        # Act
        n_sd = sut.SD_num

        # Assert
        assert sut['volume'].shape == sut['n'].shape
        assert sut.SD_num == (n != 0).sum()
        assert sut['n'].sum() == n.sum()
        assert (sut['volume'] * sut['n']).sum() == (volume * n).sum()

    @staticmethod
    @pytest.fixture(params=[1, 2, 3, 4, 5, 8, 9, 16])
    def thread_number(request):
        return request.param

    @pytest.mark.parametrize('n, cells, n_sd, idx, new_idx, cell_start', [
        ([1, 1, 1], [2, 0, 1], 3, [2, 0, 1], [1, 2, 0], [0, 1, 2, 3]),
        ([0, 1, 0, 1, 1], [3, 4, 0, 1, 2], 3, [4, 1, 3, 2, 0], [3, 4, 1], [0, 0, 1, 2, 2, 3]),
        ([1, 2, 3, 4, 5, 6, 0], [2, 2, 2, 2, 1, 1, 1], 6, [0, 1, 2, 3, 4, 5, 6], [4, 5, 0, 1, 2, 3], [0, 0, 2, 6])
    ])
    def test_sort_by_cell_id(self, n, cells, n_sd, idx, new_idx, cell_start, thread_number):
        # Arrange
        particles = DummyParticles(backend, n_sd=n_sd)
        sut = TestableStateFactory.empty_state(particles)
        sut.n = TestState.storage(n)
        n_cell = max(cells) + 1
        sut.cell_id = TestState.storage(cells)
        sut._State__idx = TestState.storage(idx)
        idx_length = len(sut._State__idx)
        sut._State__cell_start = TestState.storage([0] * (n_cell + 1))
        sut._State__n_sd = particles.n_sd
        sut._State__cell_caretaker = backend.make_cell_caretaker(sut._State__idx, sut._State__cell_start)

        # Act
        sut._State__sort_by_cell_id()

        # Assert
        assert len(sut._State__idx) == idx_length
        np.testing.assert_array_equal(np.array(new_idx), backend.to_ndarray(sut._State__idx[:sut.SD_num]))
        np.testing.assert_array_equal(np.array(cell_start), backend.to_ndarray(sut._State__cell_start))

    def test_recalculate_cell_id(self):
        # Arrange
        n = np.ones(1, dtype=np.int64)
        droplet_id = 0
        initial_position = Default.from_ndarray(np.array([[0, 0]]))
        grid = (1, 1)
        particles = DummyParticles(backend, n_sd=1)
        particles.set_environment(DummyEnvironment, {'grid': grid})
        sut = TestableStateFactory.state_2d(n=n, intensive={}, extensive={},
                                            particles=particles, positions=initial_position)
        sut.cell_origin[droplet_id, 0] = .1
        sut.cell_origin[droplet_id, 1] = .2
        sut.cell_id[droplet_id] = -1

        # Act
        sut.recalculate_cell_id()

        # Assert
        assert sut.cell_id[droplet_id] == 0

    def test_permutation_global(self):
        n_sd = 8
        idx = range(n_sd)
        u01 = [.1, .4, .2, .5, .9, .1, .6, .3]

        # Arrange
        particles = DummyParticles(backend, n_sd=n_sd)
        sut = TestableStateFactory.empty_state(particles)
        sut._State__idx = TestState.storage(idx)
        idx_length = len(sut._State__idx)
        sut._State__tmp_idx = TestState.storage([0] * idx_length)
        sut._State__sorted = True
        sut._State__n_sd = particles.n_sd
        u01 = TestState.storage(u01)

        # Act
        sut.permutation_global(u01)

        # Assert
        expected = np.array([1, 3, 5, 7, 6, 0, 4, 2])
        np.testing.assert_array_equal(sut._State__idx, expected)
        assert sut._State__sorted == False

    def test_permutation_local(self):
        n_sd = 8
        idx = range(n_sd)
        u01 = [.1, .4, .2, .5, .9, .1, .6, .3]
        cell_start = [0, 0, 2, 5, 7, n_sd]

        # Arrange
        particles = DummyParticles(backend, n_sd=n_sd)
        sut = TestableStateFactory.empty_state(particles)
        sut._State__idx = TestState.storage(idx)
        idx_length = len(sut._State__idx)
        sut._State__tmp_idx = TestState.storage([0] * idx_length)
        sut._State__cell_start = TestState.storage(cell_start)
        sut._State__sorted = True
        sut._State__n_sd = particles.n_sd
        u01 = TestState.storage(u01)

        # Act
        sut.permutation_local(u01)

        # Assert
        expected = np.array([1, 0, 2, 3, 4, 5, 6, 7])
        np.testing.assert_array_equal(sut._State__idx, expected)
        assert sut._State__sorted == True

    def test_permutation_global_repeatable(self):
        n_sd = 800
        idx = range(n_sd)
        u01 = np.random.random(n_sd)

        # Arrange
        particles = DummyParticles(backend, n_sd=n_sd)
        sut = TestableStateFactory.empty_state(particles)
        sut._State__idx = TestState.storage(idx)
        idx_length = len(sut._State__idx)
        sut._State__tmp_idx = TestState.storage([0] * idx_length)
        sut._State__sorted = True
        sut._State__n_sd = particles.n_sd
        u01 = TestState.storage(u01)

        # Act
        sut.permutation_global(u01)
        expected = backend.to_ndarray(sut._State__idx)
        sut._State__sorted = True
        sut._State__idx = TestState.storage(idx)
        sut.permutation_global(u01)

        # Assert
        np.testing.assert_array_equal(sut._State__idx, expected)
        assert sut._State__sorted == False

    def test_permutation_local_repeatable(self):
        n_sd = 800
        idx = range(n_sd)
        u01 = np.random.random(n_sd)
        cell_start = [0, 0, 20, 250, 700, n_sd]

        # Arrange
        particles = DummyParticles(backend, n_sd=n_sd)
        sut = TestableStateFactory.empty_state(particles)
        sut._State__idx = TestState.storage(idx)
        idx_length = len(sut._State__idx)
        sut._State__tmp_idx = TestState.storage([0] * idx_length)
        cell_id = []
        for i in range(len(cell_start) - 1):
            cell_id = cell_id + [i] * cell_start[i+1]
        sut.cell_id = TestState.storage(cell_id)
        sut._State__cell_start = TestState.storage(cell_start)
        sut._State__sorted = True
        sut._State__n_sd = particles.n_sd
        u01 = TestState.storage(u01)

        # Act
        sut.permutation_local(u01)
        expected = backend.to_ndarray(sut._State__idx)
        sut._State__idx = TestState.storage(idx)
        sut.permutation_local(u01)

        # Assert
        np.testing.assert_array_equal(sut._State__idx, expected)
        assert sut._State__sorted == True
        sut._State__sort_by_cell_id()
        np.testing.assert_array_equal(sut._State__idx[:50], expected[:50])
