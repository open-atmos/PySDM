"""
Created at 31.07.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import pytest
import numpy as np

from PySDM.backends.default import Default
from PySDM.backends.numba.numba import Numba
from PySDM.backends.thrustRTC.thrustRTC import ThrustRTC

# noinspection PyUnresolvedReferences
from tests.unit_tests.backends.__parametrisation__ import shape_full, shape_1d, shape_2d, \
                                               dtype_full, dtype, \
                                               length, natural_length, \
                                               order

backend = Default()


@pytest.mark.parametrize('sut', [Numba(), ThrustRTC(), ])  # Pythran()])
class TestBackend:
    @staticmethod
    def data(sut_backend, shape, dtype, seed=0):
        np.random.seed(seed)
        rand_ndarray = (100*np.random.rand(*shape)).astype(dtype)

        result_sut = sut_backend.from_ndarray(rand_ndarray)
        result_default = backend.from_ndarray(rand_ndarray)

        return result_sut, result_default

    @staticmethod
    def idx_length(shape):
        if len(shape) >= 2:
            result = shape[1]
        else:
            result = shape[0]

        return result

    @staticmethod
    def idx(sut_backend, shape, order, seed=0):
        np.random.seed(seed)

        idx_len = TestBackend.idx_length(shape)

        idx_ndarray = np.arange(idx_len)

        if order == 'desc':
            idx_ndarray = idx_ndarray[::-1]
        elif order == 'random':
            np.random.permutation(idx_ndarray)

        result_sut = sut_backend.from_ndarray(idx_ndarray)
        result_default = backend.from_ndarray(idx_ndarray)

        return result_sut, result_default

    @staticmethod
    def length(length, shape):
        idx_len = TestBackend.idx_length(shape)

        if length == 'zero':
            return 0
        elif length == 'middle':
            return (idx_len + 1) // 2
        elif length == 'full':
            return idx_len

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
    def test_array(sut, dtype_full, shape_full):
        try:
            # Act
            actual = sut.array(shape_full, dtype_full)
            expected = backend.array(shape_full, dtype_full)

            # Assert
            assert sut.shape(actual) == backend.shape(expected)
            assert sut.dtype(actual) == backend.dtype(expected)
        except NotImplementedError:
            if dtype_full in (float, int):
                assert False

    @staticmethod
    def test_from_ndarray(sut, dtype_full, shape_full):
        # Arrange
        ndarray = np.empty(shape=shape_full, dtype=dtype_full)

        try:
            # Act
            actual = sut.from_ndarray(ndarray)
            expected = backend.from_ndarray(ndarray)

            # Assert
            np.testing.assert_array_equal(sut.to_ndarray(actual), backend.to_ndarray(expected))
        except NotImplementedError:
            if dtype_full in (float, int):
                assert False

    # TODO test_write_row()

    # TODO test_read_row()

    # TODO idx as input
    @staticmethod
    def test_shuffle(sut, shape_1d, natural_length):
        # Arrange
        axis = 0
        sut_data, data = TestBackend.data(sut, shape_1d, int)
        sut_idx, idx = TestBackend.idx(sut, shape_1d, 'asc')
        length = TestBackend.length(natural_length, shape_1d)
        # Act
        sut.shuffle(sut_data, length, axis)
        backend.shuffle(data, length, axis)

        # Assert
        sut_data_original, data_original = TestBackend.data(sut, shape_1d, int)
        assert sut.shape(sut_data) == backend.shape(data)
        assert sut.amin(sut_data, sut_idx, length) == sut.amin(sut_data_original, sut_idx, length)
        assert sut.amax(sut_data, sut_idx, length) == sut.amax(sut_data_original, sut_idx, length)

    @staticmethod
    def test_stable_argsort(sut, shape_1d, length, order):
        # Arrange
        sut_data, data = TestBackend.data(sut, shape_1d, int)
        sut_idx, idx = TestBackend.idx(sut, shape_1d, order)
        length = TestBackend.length(length, shape_1d)

        # Act
        sut.stable_argsort(sut_idx, sut_data, length)
        backend.stable_argsort(idx, data, length)

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
    def test_shape(sut, shape_full, dtype):
        # Arrange
        sut_data, data = TestBackend.data(sut, shape_full, dtype)

        # Act
        actual = sut.shape(sut_data)
        expected = backend.shape(data)

        # Assert
        assert actual == expected

    @staticmethod
    def test_dtype(sut, shape_full, dtype):
        # Arrange
        sut_data, data = TestBackend.data(sut, shape_full, dtype)

        # Act
        actual = sut.dtype(sut_data)
        expected = backend.dtype(data)

        # Assert
        assert actual == expected

    @staticmethod
    def test_urand(sut, shape_1d):
        # Arrange
        sut_data, data = TestBackend.data(sut, shape_1d, float)
        sut_idx, idx = TestBackend.idx(sut, shape_1d, 'asc')
        length = shape_1d[0]

        # Act
        sut.urand(sut_data)
        backend.urand(data)

        # Assert
        assert sut.shape(sut_data) == backend.shape(data)
        assert sut.dtype(sut_data) == backend.dtype(data)
        assert sut.amin(sut_data, sut_idx, length) >= 0
        assert sut.amax(sut_data, sut_idx, length) <= 1

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

    @staticmethod
    @pytest.mark.parametrize('multiplier', [0., 1., 87., -5., .7, -.44])
    def test_multiply_scalar(sut, shape_1d, multiplier):
        # Arrange
        sut_data, data = TestBackend.data(sut, shape_1d, float)

        # Act
        sut.multiply(sut_data, multiplier)
        backend.multiply(data, multiplier)

        # Assert
        np.testing.assert_array_equal(sut.to_ndarray(sut_data), backend.to_ndarray(data))

    @staticmethod
    def test_multiply_elementwise(sut, shape_1d):
        # Arrange
        sut_data, data = TestBackend.data(sut, shape_1d, float)
        sut_multiplier, multiplier = TestBackend.data(sut, shape_1d, float, seed=1)

        # Act
        sut.multiply(sut_data, sut_multiplier)
        backend.multiply(data, multiplier)

        # Assert
        np.testing.assert_array_equal(sut.to_ndarray(sut_data), backend.to_ndarray(data))
        np.testing.assert_array_equal(sut.to_ndarray(sut_multiplier), backend.to_ndarray(multiplier))

    @staticmethod
    def test_sum(sut, shape_1d):
        # Arrange
        sut_data, data = TestBackend.data(sut, shape_1d, float)
        sut_data_in, data_in = TestBackend.data(sut, shape_1d, float)

        # Act
        sut.add(sut_data, sut_data_in)
        backend.add(data, data_in)

        # Assert
        np.testing.assert_array_equal(sut.to_ndarray(sut_data), backend.to_ndarray(data))

    @staticmethod
    # TODO data<0
    def test_floor(sut, shape_1d):
        # Arrange
        sut_data, data = TestBackend.data(sut, shape_1d, float)

        # Act
        sut.floor_in_place(sut_data)
        backend.floor_in_place(data)

        # Assert
        np.testing.assert_array_equal(sut.to_ndarray(sut_data), backend.to_ndarray(data))

    @staticmethod
    def test_to_ndarray(sut, shape_full, dtype):
        # Arrange
        sut_data, data = TestBackend.data(sut, shape_full, dtype)

        # Act
        actual = sut.to_ndarray(sut_data)
        expected = backend.to_ndarray(data)

        # Assert
        np.testing.assert_array_equal(actual, expected)
