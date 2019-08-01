"""
Created at 31.07.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import pytest
import numpy as np

from SDM.backends.default import Default
from SDM.backends.numpy import Numpy
from SDM.backends.numba import Numba
from SDM.backends.numba_parallel import NumbaParallel


@pytest.fixture(params=[
    pytest.param((1,)),
    pytest.param((8,)),
    pytest.param((1, 1)),
    pytest.param((1, 8)),
    pytest.param((8, 1)),
    pytest.param((8, 8)),
])
def shape_full(request):
    return request.param


@pytest.fixture(params=[
    pytest.param((1, 1)),
    pytest.param((1, 8)),
    pytest.param((8, 1)),
    pytest.param((8, 8)),
])
def shape(request):
    return request.param


@pytest.fixture(params=[
    pytest.param(float),
    pytest.param(int),
    pytest.param(bytes),
])
def dtype_full(request):
    return request.param


@pytest.fixture(params=[
    pytest.param(float),
    pytest.param(int),
])
def dtype(request):
    return request.param


@pytest.fixture(params=[
    pytest.param('zero'),
    pytest.param('middle'),
    pytest.param('full'),
])
def length(request):
    return request.param


@pytest.mark.parametrize('sut', [Numpy, Numba, NumbaParallel])
class TestBackend:
    @staticmethod
    def data(backend, shape, dtype, seed=0):
        np.random.seed(seed)
        rand_ndarray = np.random.rand(*shape).astype(dtype)

        result_sut = backend.from_ndarray(rand_ndarray)
        result_default = Default.from_ndarray(rand_ndarray)

        return result_sut, result_default

    @staticmethod
    def idx(backend, shape):
        idx_ndarray = np.arange(shape[1])

        result_sut = backend.from_ndarray(idx_ndarray)
        result_default = Default.from_ndarray(idx_ndarray)

        return result_sut, result_default

    @staticmethod
    def length(length, shape):
        if length == 'zero':
            return 0
        elif length == 'middle':
            return shape[0] // 2
        elif length == 'full':
            return shape[0]

    def test_array(self, sut, dtype_full, shape_full):
        try:
            actual = sut.array(shape_full, dtype_full)
            expected = Default.array(shape_full, dtype_full)

            assert sut.shape(actual) == Default.shape(expected)
            assert sut.dtype(actual) == Default.dtype(actual)
        except NotImplementedError:
            if dtype_full in (float, int):
                assert False

    def test_from_ndarray(self, sut, dtype_full, shape_full):
        # Arrange
        ndarray = np.empty(shape=shape_full, dtype=dtype_full)

        try:
            # Act
            actual = sut.from_ndarray(ndarray)
            expected = Default.from_ndarray(ndarray)

            # Assert
            np.testing.assert_array_equal(sut.to_ndarray(actual), Default.to_ndarray(expected))
        except NotImplementedError:
            if dtype_full in (float, int):
                assert False

    # TODO require backend.seed(seed=)
    @pytest.mark.parametrize('axis', [0, 1])
    def test_shuffle(self, sut, dtype, shape, length, axis):
        sut_data, data = TestBackend.data(sut, shape, dtype)
        sut_idx, idx = TestBackend.idx(sut, shape)
        # idx = np.random.permutation(length)
        # Numpy.reindex(data, idx, length, axis=axis)

    # @staticmethod
    # def reindex(data, idx, length, axis):
    #     if axis == 1:
    #         data[:, 0:length] = data[:, idx]
    #     else:
    #         raise NotImplementedError

    def test_argsort(self, sut, shape, dtype, length):
        # Arrange
        sut_data, data = TestBackend.data(sut, shape, dtype)
        sut_idx, idx = TestBackend.idx(sut, shape)
        length = TestBackend.length(length, shape)

        # Act
        sut.argsort(sut_data, sut_idx, length)
        Default.argsort(data, idx, length)

        # Assert
        np.testing.assert_array_equal(sut.to_ndarray(sut_data), Default.to_ndarray(data))
        np.testing.assert_array_equal(sut.to_ndarray(sut_idx), Default.to_ndarray(idx))

    def test_stable_argsort(self, sut, shape, dtype, length):
        # Arrange
        sut_data, data = TestBackend.data(sut, shape, dtype)
        sut_idx, idx = TestBackend.idx(sut, shape)
        length = TestBackend.length(length, shape)

        # Act
        sut.stable_argsort(sut_data, sut_idx, length)
        Default.stable_argsort(data, idx, length)

        # Assert
        np.testing.assert_array_equal(sut.to_ndarray(sut_data), Default.to_ndarray(data))
        np.testing.assert_array_equal(sut.to_ndarray(sut_idx), Default.to_ndarray(idx))

    def test_amin(self, sut, shape, dtype):
        # Arrange
        sut_data, data = TestBackend.data(sut, shape, dtype)

        # Act
        actual = sut.amin(sut_data)
        expected = Default.amin(data)

        # Assert
        assert actual == expected

    def test_amax(self, sut, shape, dtype):
        # Arrange
        sut_data, data = TestBackend.data(sut, shape, dtype)

        # Act
        actual = sut.amax(sut_data)
        expected = Default.amax(data)

        # Assert
        assert actual == expected

    # @staticmethod
    # def transform(data, func, length):
    #     data[:length] = np.fromfunction(
    #         np.vectorize(func, otypes=(data.dtype,)),
    #         (length,),
    #         dtype=np.int
    #     )
    #
    # @staticmethod
    # def foreach(data, func):
    #     for i in range(len(data)):
    #         func(i)
    #

    def test_shape(self, sut, shape, dtype):
        # Arrange
        sut_data, data = TestBackend.data(sut, shape, dtype)

        # Act
        actual = sut.shape(sut_data)
        expected = Default.shape(data)

        # Assert
        assert actual == expected

    def test_dtype(self, sut, shape, dtype):
        # Arrange
        sut_data, data = TestBackend.data(sut, shape, dtype)

        # Act
        actual = sut.dtype(sut_data)
        expected = Default.dtype(data)

        # Assert
        assert actual == expected
    #
    # @staticmethod
    # def urand(data, min=0, max=1):
    #     data[:] = np.random.uniform(min, max, data.shape)
    #
    # # TODO do not create array
    # @staticmethod
    # def remove_zeros(data, idx, length) -> int:
    #     for i in range(length):
    #         if data[0][idx[0][i]] == 0:
    #             idx[0][i] = idx.shape[1]
    #     idx.sort()
    #     return np.count_nonzero(data)
    #
    # @staticmethod
    # def extensive_attr_coalescence(n, idx, length, data, gamma):
    #     # TODO in segments
    #     for i in range(length // 2):
    #         j = 2 * i
    #         k = j + 1
    #
    #         j = idx[j]
    #         k = idx[k]
    #
    #         if n[j] < n[k]:
    #             j, k = k, j
    #         g = min(gamma[i], n[j] // n[k])
    #
    #         new_n = n[j] - g * n[k]
    #         if new_n > 0:
    #             data[:, k] += g * data[:, j]
    #         else:  # new_n == 0
    #             data[:, j] = g * data[:, j] + data[:, k]
    #             data[:, k] = data[:, j]
    #
    # @staticmethod
    # def n_coalescence(n, idx, length, gamma):
    #     # TODO in segments
    #     for i in range(length // 2):
    #         j = 2 * i
    #         k = j + 1
    #
    #         j = idx[j]
    #         k = idx[k]
    #
    #         if n[j] < n[k]:
    #             j, k = k, j
    #         g = min(gamma[i], n[j] // n[k])
    #
    #         new_n = n[j] - g * n[k]
    #         if new_n > 0:
    #             n[j] = new_n
    #         else:  # new_n == 0
    #             n[j] = n[k] // 2
    #             n[k] = n[k] - n[j]
    #
    # @staticmethod
    # def sum_pair(data_out, data_in, idx, length):
    #     for i in range(length // 2):
    #         data_out[i] = data_in[idx[2 * i]] + data_in[idx[2 * i + 1]]

    # @staticmethod
    # def max_pair(data_out, data_in, idx, length):
    #     for i in range(length // 2):
    #         data_out[i] = max(data_in[idx[2 * i]], data_in[idx[2 * i + 1]])
    #
    #
    # def test_max_pair(self, sut, dtype, length):
    #     # Arrange
    #     shape = (1, 87)
    #     sut_data, data = TestBackend.data(sut, shape, dtype)
    #     sut_data_in, data_in = TestBackend.data(sut, shape, dtype, seed=1)
    #     sut_idx, idx = TestBackend.idx(sut, shape)
    #     length = TestBackend.length(length, shape)
    #
    #     # Act
    #     sut.max_pair(sut_data, sut_data_in, sut_idx, length)
    #     Default.max_pair(data, data_in, idx, length)
    #
    #     # Assert
    #     np.testing.assert_array_equal(sut.to_ndarray(sut_data), Default.to_ndarray(data))
    #     np.testing.assert_array_equal(sut.to_ndarray(sut_data_in), Default.to_ndarray(data_in))
    #     np.testing.assert_array_equal(sut.to_ndarray(sut_idx), Default.to_ndarray(idx))

    @pytest.mark.parametrize('multiplier', [0, 1, 2, 87, -5])
    def test_multiply_scalar(self, sut, shape, dtype, multiplier):
        # Arrange
        sut_data, data = TestBackend.data(sut, shape, dtype)

        # Act
        sut.multiply(sut_data, multiplier)
        Default.multiply(data, multiplier)

        # Assert
        np.testing.assert_array_equal(sut.to_ndarray(sut_data), Default.to_ndarray(data))

    # TODO
    @pytest.mark.xfail
    def test_multiply_elementwise(self, sut, shape, dtype, multiplier):
        assert False

    # @staticmethod
    # def sum(data_out, data_in):
    #     data_out[:] = data_out + data_in
    #
    def test_floor(self, sut, shape, dtype):
        # Arrange
        sut_data, data = TestBackend.data(sut, shape, dtype)

        # Act
        sut.floor(sut_data)
        Default.floor(data)

        # Assert
        np.testing.assert_array_equal(sut.to_ndarray(sut_data), Default.to_ndarray(data))

    # @staticmethod
    # def to_ndarray(data):
    #     return data
