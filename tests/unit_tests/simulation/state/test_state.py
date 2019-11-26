from tests.unit_tests.simulation.state.testable_state_factory import TestableStateFactory
from tests.unit_tests.simulation.state.dummy_particles import DummyParticles
from PySDM.backends.default import Default

import numpy as np
import pytest

backend = Default


class TestState:

    @staticmethod
    def storage(iterable):
        return backend.from_ndarray(np.array(iterable))

    @staticmethod
    def check_contiguity(state, attr='n', i_SD=0):
        item = state[attr]
        assert item.flags['C_CONTIGUOUS']
        assert item.flags['F_CONTIGUOUS']

        sd = state.get_SD(i_SD)
        assert not sd.flags['C_CONTIGUOUS']
        assert not sd.flags['F_CONTIGUOUS']

    @pytest.mark.xfail
    def test_get_item_does_not_copy(self):
        # Arrange
        arr = np.ones(10)
        sut = State({'n': arr, 'a': arr, 'b': arr}, backend=backend)

        # Act
        item = sut['a']

        # Assert
        assert item.base.__array_interface__['data'] == sut.data.__array_interface__['data']

    @pytest.mark.xfail
    def test_get_sd_does_not_copy(self):
        # Arrange
        arr = np.ones(10)
        sut = State({'n': arr, 'a': arr, 'b': arr}, backend=backend)

        # Act
        item = sut.get_SD(5)

        # Assert
        assert item.base.__array_interface__['data'] == sut.data.__array_interface__['data']

    @pytest.mark.xfail
    def test_contiguity(self):
        # Arrange
        arr = np.ones(10)
        sut = State({'n': arr, 'a': arr, 'b': arr}, backend=backend)

        # Act & Assert
        self.check_contiguity(sut, attr='a', i_SD=5)

    @pytest.mark.parametrize("x, n", [
        pytest.param(np.array([1., 1, 1, 1]), np.array([1, 1, 1, 1])),
        pytest.param(np.array([1., 2, 1, 1]), np.array([2, 0, 2, 0])),
        pytest.param(np.array([1., 1, 4]), np.array([5, 0, 0]))
    ])
    def test_housekeeping(self, x, n):
        # Arrange
        particles = DummyParticles(backend, n_sd=len(n))
        sut = TestableStateFactory.state_0d(n=n, extensive={'x': x}, intensive={}, particles=particles)
        # TODO
        sut.healthy = sut.backend.from_ndarray(np.array([0]))

        # Act
        sut.housekeeping()

        # Assert
        assert sut['x'].shape == sut['n'].shape
        assert sut.SD_num == (n != 0).sum()
        assert sut['n'].sum() == n.sum()
        assert (sut['x'] * sut['n']).sum() == (x * n).sum()

    def test_sort_by_cell_id(self):
        # Arrange
        particles = DummyParticles(backend, n_sd=3)
        sut = TestableStateFactory.empty_state(particles)
        sut.n = TestState.storage([0, 1, 0, 1, 1])
        sut.cell_id = TestState.storage([3, 4, 0, 1, 2])
        sut.idx = TestState.storage([4, 1, 3, 2, 0])
        sut.SD_num = particles.n_sd

        # Act
        sut.sort_by_cell_id()

        # Assert
        np.testing.assert_array_equal(np.array([3, 4, 1, 2, 0]), backend.to_ndarray(sut.idx))

    def test_recalculate_cell_id(self):
        # Arrange
        n = np.ones(1)
        droplet_id = 0
        initial_position = Default.from_ndarray(np.array([[0, 0]]))
        particles= DummyParticles(backend, n_sd = 1)
        sut = TestableStateFactory.state_2d(n=n, grid=(1, 1), intensive={}, extensive={},
                                            particles=particles, positions=initial_position)
        sut.cell_origin[droplet_id, 0] = .1
        sut.cell_origin[droplet_id, 1] = .2
        sut.cell_id[droplet_id] = -1

        # Act
        sut.recalculate_cell_id()

        # Assert
        assert sut.cell_id[droplet_id] == 0


