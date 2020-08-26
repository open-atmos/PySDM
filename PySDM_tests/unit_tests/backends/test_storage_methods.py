"""
Created at 21.03.2020
"""

import pytest
import numpy as np
# noinspection PyUnresolvedReferences
from PySDM_tests.unit_tests.backends.__parametrisation__ import \
    shape_full, shape_2d, shape_1d, \
    dtype_full, dtype, \
    length
from PySDM_tests.unit_tests.backends.__parametrisation__ import backend, backends
from .utils import universal_test, generate_data


# TODO: not implemented
@pytest.mark.skip()
@pytest.mark.parametrize('sut', backends)
class TestStorageMethods:

    @staticmethod
    def test_array(sut, shape_full, dtype_full):
        try:
            # Act
            actual = sut.array(shape_full, dtype_full)
            expected = backend.array(shape_full, dtype_full)

            # Assert
            assert actual.shape == expected.shape
            assert actual.dtype == expected.dtype
        except NotImplementedError:
            if dtype_full in (float, int):
                assert False

    @staticmethod
    def test_download(sut, shape_full, dtype):
        # Arrange
        sut_backend_data, backend_data = generate_data(sut, shape_full, dtype)
        if dtype is int:
            np_dtype = np.int64
        elif dtype is float:
            np_dtype = np.float64
        else:
            raise NotImplementedError()
        numpy_target = np.empty(shape_full, np_dtype)
        sut_numpy_target = np.empty(shape_full, np_dtype)

        # Act
        backend.download(backend_data, numpy_target)
        sut.download(sut_backend_data, sut_numpy_target)

        # Assert
        np.testing.assert_array_equal(sut_numpy_target, numpy_target)
        np.testing.assert_array_equal(sut.to_ndarray(sut_backend_data), backend.to_ndarray(backend_data))

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

    @staticmethod
    def test_range(sut, shape_full):
        for start in (0, shape_full[0] // 2, shape_full[0]):
            for stop in (shape_full[0] // 2, shape_full[0] - 1, shape_full[0]):
                if stop < start:
                    continue
                params = [{'name': "array",
                           'details': {'shape': shape_full}},
                          {'name': "start",
                           'details': {'value': start}},
                          {'name': "stop",
                           'details': {'value': stop}}
                          ]
                universal_test("range", sut, params)

    @staticmethod
    def test_read_row(sut, shape_2d):
        for i in range(shape_2d[0]-1):
            params = [{'name': "array",
                       'details': {'shape': shape_2d, 'dtype': float}},
                      {'name': "i",
                       'details': {'value': i}}
                      ]
            universal_test("read_row", sut, params)

    @staticmethod
    def test_shuffle_global(sut, shape_1d, length):
        assert False
        # params = [{'name': "idx",
        #            'details': {'shape': shape_1d}},
        #           {'name': "length",
        #            'details': {'length': length, 'shape': shape_1d}},
        #           {'name': "u01",
        #            'details': {'shape': shape_1d, 'dtype': float, 'factor': 1}}
        #           ]
        # universal_test("shuffle_global", sut, params)

    @staticmethod
    def test_shuffle_local(sut, shape_1d, length):
        assert False

    @staticmethod
    def test_to_ndarray(sut, shape_full, dtype):
        # Arrange
        sut_data, data = generate_data(sut, shape_full, dtype)

        # Act
        actual = sut.to_ndarray(sut_data)
        expected = backend.to_ndarray(data)

        # Assert
        np.testing.assert_array_equal(actual, expected)

    @staticmethod
    def test_upload(sut, shape_full, dtype):
        # Arrange
        np.random.seed(44)
        numpy_data = np.random.rand(*shape_full).astype(dtype)
        sut_numpy_data = np.empty(shape_full)
        sut_numpy_data[:] = numpy_data[:]
        sut_backend_target, backend_target = generate_data(sut, shape_full, dtype)

        # Act
        backend.upload(numpy_data, backend_target)
        sut.upload(sut_numpy_data, sut_backend_target)

        # Assert
        np.testing.assert_array_equal(sut.to_ndarray(sut_backend_target), backend.to_ndarray(backend_target))
        np.testing.assert_array_equal(sut_numpy_data, numpy_data)

    @staticmethod
    def test_write_row(sut, shape_2d):
        for i in range(shape_2d[0]-1):
            params = [{'name': "array",
                       'details': {'shape': shape_2d, 'dtype': float}},
                      {'name': "i",
                       'details': {'value': i}},
                      {'name': "row",
                       'details': {'shape': (shape_2d[1],), 'dtype': float}}
                      ]
            universal_test("write_row", sut, params)
