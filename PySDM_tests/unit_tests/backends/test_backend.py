"""
Created at 31.07.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import pytest
import numpy as np

# noinspection PyUnresolvedReferences
from PySDM_tests.unit_tests.backends.__parametrisation__ import shape_full, shape_1d, shape_2d, \
                                               dtype_full, dtype, \
                                               length, natural_length, \
                                               order
from PySDM_tests.unit_tests.backends.__parametrisation__ import backend, backends


@pytest.mark.parametrize('sut', backends)
class TestBackend:

    @staticmethod
    def is_first_in_pair(sut_backend, length, seed=0):
        np.random.seed(seed)

        is_first_in_pair = np.random.randint(2, size=length)
        pair = False
        for i in range(length):
            if pair:
                is_first_in_pair[i] = 0
                pair = False
            elif is_first_in_pair[i] == 1:
                pair = True
        sut_is_first_in_pair = sut_backend.from_ndarray(is_first_in_pair)
        backend_is_first_in_pair = backend.from_ndarray(is_first_in_pair)

        return sut_is_first_in_pair, backend_is_first_in_pair

    @staticmethod
    @pytest.mark.xfail
    def test_stable_argsort(sut, shape_1d, length, order):
        # Arrange
        sut_data, data = TestBackend.data(sut, shape_1d, int)
        sut_idx, idx = TestBackend.idx(sut, shape_1d, order)
        length = TestBackend.length(length, shape_1d)

        # Act
        sut.counting_sort_by_cell_id(sut_idx, sut_data, length)
        backend.counting_sort_by_cell_id(idx, data, length)

        # Assert
        np.testing.assert_array_equal(sut.to_ndarray(sut_data), backend.to_ndarray(data))
        np.testing.assert_array_equal(sut.to_ndarray(sut_idx), backend.to_ndarray(idx))

    @staticmethod
    def test_amin(sut, shape_1d, natural_length, order):
        # Arrange
        sut_data, data = TestBackend.data(sut, shape_1d, float)
        sut_idx, idx = TestBackend.idx(sut, shape_1d, order)
        length = TestBackend.length(natural_length, shape_1d)

        # Act
        actual = sut.amin(sut_data, sut_idx, length)
        expected = backend.amin(data, idx, length)

        # Assert
        assert actual == expected

    @staticmethod
    def test_amax(sut, shape_1d, natural_length, order):
        # Arrange
        sut_data, data = TestBackend.data(sut, shape_1d, float)
        sut_idx, idx = TestBackend.idx(sut, shape_1d, order)
        length = TestBackend.length(natural_length, shape_1d)

        # Act
        actual = sut.amax(sut_data, sut_idx, length)
        expected = backend.amax(data, idx, length)

        # Assert
        assert actual == expected

    @staticmethod
    @pytest.mark.parametrize('data_ndarray', [
        np.array([0] * 87),
        np.array([1, 0, 1, 0, 1, 1, 1, 1]),
        np.array([1, 1, 1, 1, 1, 0, 1, 0]),
        np.array([1] * 87)
    ])
    def test_remove_zeros(sut, data_ndarray, length, order):
        # Arrange
        shape = data_ndarray.shape
        sut_data = sut.from_ndarray(data_ndarray)
        data = backend.from_ndarray(data_ndarray)
        sut_idx, idx = TestBackend.idx(sut, shape, order)
        length = TestBackend.length(length, shape)

        # Act
        sut_new_length = sut.remove_zeros(sut_data, sut_idx, length)
        new_length = backend.remove_zeros(data, idx, length)

        # Assert
        assert sut_new_length == new_length
        np.testing.assert_array_equal(sut.to_ndarray(sut_data), backend.to_ndarray(data))
        np.testing.assert_array_equal(sut.to_ndarray(sut_idx)[sut_new_length:], backend.to_ndarray(idx)[new_length:])
        np.testing.assert_array_equal(sut.to_ndarray(sut_idx)[:sut_new_length].sort(),
                                      sut.to_ndarray(sut_idx)[:new_length].sort())

    @staticmethod
    # TODO new_n == 0
    def test_coalescence(sut, shape_2d, natural_length, order):
        # Arrange
        sut_n, n = TestBackend.data(sut, (shape_2d[1],), int)
        sut_data, data = TestBackend.data(sut, shape_2d, float)
        sut_idx, idx = TestBackend.idx(sut, shape_2d, order)
        length = TestBackend.length(natural_length, shape_2d)
        sut_healthy = sut.from_ndarray(np.array([1]))
        healthy = backend.from_ndarray(np.array([1]))

        assert backend.amin(n, idx, length) > 0

        # TODO insert 0 in odd position in gamma array  (per pairs)
        sut_gamma = sut.from_ndarray(np.arange(shape_2d[1]).astype(np.float64))
        gamma = backend.from_ndarray(np.arange(shape_2d[1]).astype(np.float64))

        # Act
        # TODO intensive
        sut.coalescence(sut_n, sut_idx, length, sut_data, sut_data, sut_gamma, sut_healthy)
        backend.coalescence(n, idx, length, data, data, gamma, healthy)

        # Assert
        np.testing.assert_array_equal(sut.to_ndarray(sut_n), backend.to_ndarray(n))
        np.testing.assert_array_equal(sut.to_ndarray(sut_n), backend.to_ndarray(n))
        np.testing.assert_array_equal(sut.to_ndarray(sut_idx), backend.to_ndarray(idx))
        np.testing.assert_array_equal(sut.to_ndarray(sut_data), backend.to_ndarray(data))
        np.testing.assert_array_equal(sut.to_ndarray(sut_gamma), backend.to_ndarray(gamma))
        np.testing.assert_array_equal(sut.to_ndarray(sut_healthy), backend.to_ndarray(healthy))

    @staticmethod
    def test_sum_pair(sut, shape_1d, length, order):
        # Arrange
        sut_data, data = TestBackend.data(sut, shape_1d, float)
        sut_data_in, data_in = TestBackend.data(sut, shape_1d, float, seed=1)
        sut_idx, idx = TestBackend.idx(sut, shape_1d, order)
        length = TestBackend.length(length, shape_1d)
        sut_is_first_in_pair, is_first_in_pair = TestBackend.is_first_in_pair(sut, length)

        # Act
        sut.sum_pair(sut_data, sut_data_in, sut_is_first_in_pair, sut_idx, length)
        backend.sum_pair(data, data_in, is_first_in_pair, idx, length)

        # Assert
        np.testing.assert_array_equal(sut.to_ndarray(sut_data), backend.to_ndarray(data))
        np.testing.assert_array_equal(sut.to_ndarray(sut_data_in), backend.to_ndarray(data_in))
        np.testing.assert_array_equal(sut.to_ndarray(sut_idx), backend.to_ndarray(idx))

    @staticmethod
    def test_max_pair(sut, shape_1d, length, order):
        # Arrange
        sut_data, data = TestBackend.data(sut, shape_1d, float)
        sut_data_in, data_in = TestBackend.data(sut, shape_1d, int, seed=1)
        sut_idx, idx = TestBackend.idx(sut, shape_1d, order)
        length = TestBackend.length(length, shape_1d)
        sut_is_first_in_pair, is_first_in_pair = TestBackend.is_first_in_pair(sut, length)

        # Act
        sut.max_pair(sut_data, sut_data_in, sut_is_first_in_pair, sut_idx, length)
        backend.max_pair(data, data_in, is_first_in_pair, idx, length)

        # Assert
        np.testing.assert_array_equal(sut.to_ndarray(sut_data), backend.to_ndarray(data))
        np.testing.assert_array_equal(sut.to_ndarray(sut_data_in), backend.to_ndarray(data_in))
        np.testing.assert_array_equal(sut.to_ndarray(sut_idx), backend.to_ndarray(idx))
