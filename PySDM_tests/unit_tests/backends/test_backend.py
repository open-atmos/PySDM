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
