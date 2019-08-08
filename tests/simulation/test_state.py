from SDM.simulation.state import State

import numpy as np
import pytest


class TestState:
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
        sut = State({'n': arr, 'a': arr, 'b': arr})

        # Act
        item = sut['a']

        # Assert
        assert item.base.__array_interface__['data'] == sut.data.__array_interface__['data']

    @pytest.mark.xfail
    def test_get_sd_does_not_copy(self):
        # Arrange
        arr = np.ones(10)
        sut = State({'n': arr, 'a': arr, 'b': arr})

        # Act
        item = sut.get_SD(5)

        # Assert
        assert item.base.__array_interface__['data'] == sut.data.__array_interface__['data']

    @pytest.mark.xfail
    def test_contiguity(self):
        # Arrange
        arr = np.ones(10)
        sut = State({'n': arr, 'a': arr, 'b': arr})

        # Act & Assert
        self.check_contiguity(sut, attr='a', i_SD=5)

    @pytest.mark.parametrize("x, n", [
        pytest.param(np.array([1., 1, 1, 1]), np.array([1, 1, 1, 1])),
        pytest.param(np.array([1., 2, 1, 1]), np.array([2, 0, 2, 0])),
        pytest.param(np.array([1., 1, 4]), np.array([5, 0, 0]))
    ])
    def test_housekeeping(self, x, n):
        # Arrange
        sut = State(n=n, extensive={'x': x}, intensive={}, segment_num=1)
        # TODO
        sut.healthy = sut.backend.from_ndarray(np.array([0]))

        # Act
        sut.housekeeping()

        # Assert
        assert sut['x'].shape == sut['n'].shape
        assert sut.SD_num == (n != 0).sum()
        assert sut['n'].sum() == n.sum()
        assert (sut['x'] * sut['n']).sum() == (x * n).sum()
