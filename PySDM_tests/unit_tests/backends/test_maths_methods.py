"""
Created at 20.03.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import pytest
import numpy as np
import warnings
# noinspection PyUnresolvedReferences
from PySDM_tests.unit_tests.backends.__parametrisation__ import \
    number_float, number_int, number, \
    shape_full, shape_1d, shape_2d, \
    dtype_full, dtype, dtype_mixed, \
    length, natural_length, \
    order
from PySDM_tests.unit_tests.backends.__parametrisation__ import backend, backends
from PySDM_tests.unit_tests.backends.test_backend import TestBackend  # TODO


@pytest.mark.parametrize('sut', backends)
class TestMathsMethods:

    @staticmethod
    def universal_test(method_name, sut, params):
        # Arrange
        default_params = {}
        sut_params = {}
        for param in params:
            sut_params[param['name']], default_params[param['name']] = \
                TestBackend.data(sut, **param['details'])
        default_method = getattr(backend, method_name)
        sut_method = getattr(sut, method_name)

        # Act
        default_method(**default_params)
        sut_method(**sut_params)

        # Assert
        for param in params:
            if 'value' in param['details'].keys():
                assert sut_params[param['name']] == default_params[param['name']]
            else:
                try:
                    np.testing.assert_array_equal(
                        sut.to_ndarray(sut_params[param['name']]),
                        backend.to_ndarray(default_params[param['name']])
                    )
                except AssertionError:
                    precision = 15
                    warnings.warn(f"Fail with high precision, try with {precision} digit precision...")
                    np.testing.assert_almost_equal(
                        sut.to_ndarray(sut_params[param['name']]),
                        backend.to_ndarray(default_params[param['name']]), decimal=precision)

    @staticmethod
    def test_add(sut, shape_full, dtype):
        params = [{'name': "output",
                  'details': {'shape': shape_full, 'dtype': dtype}},
                  {'name': "addend",
                   'details': {'shape': shape_full, 'dtype': dtype}},
                  ]
        TestMathsMethods.universal_test("add", sut, params)

    @staticmethod
    def test_column_modulo(sut, shape_2d, dtype_full):
        params = [{'name': "output",
                   'details': {'shape': shape_2d, 'dtype': int}},
                  {'name': "divisor",
                   'details': {'shape': (shape_2d[1],), 'dtype': int}},
                  ]
        TestMathsMethods.universal_test("column_modulo", sut, params)

    @staticmethod
    def test_floor(sut, shape_1d):
        params = [{'name': "output",
                   'details': {'shape': shape_1d, 'dtype': float, 'negative': True}}
                  ]
        TestMathsMethods.universal_test("floor", sut, params)

    @staticmethod
    def test_floor_out_of_place(sut, shape_2d):
        params = [{'name': "output",
                   'details': {'shape': shape_2d, 'dtype': float}},
                  {'name': "input_data",
                   'details': {'shape': shape_2d, 'dtype': float, 'negative': True}},
                  ]
        TestMathsMethods.universal_test("floor_out_of_place", sut, params)

    @staticmethod
    def test_multiply_scalar(sut, shape_full, dtype, number_float):
        params = [{'name': "output",
                   'details': {'shape': shape_full, 'dtype': dtype, 'negative': True}},
                  {'name': "multiplier",
                   'details': {'value': number_float}},
                  ]
        TestMathsMethods.universal_test("multiply", sut, params)

    @staticmethod
    def test_multiply_elementwise(sut, shape_full, dtype, dtype_mixed):
        params = [{'name': "output",
                   'details': {'shape': shape_full, 'dtype': dtype, 'negative': True}},
                  {'name': "multiplier",
                   'details': {'shape': shape_full, 'dtype': dtype_mixed, 'negative': True}},
                  ]
        TestMathsMethods.universal_test("multiply", sut, params)

    @staticmethod
    def test_multiply_out_of_place_scalar(sut, shape_full, dtype, number_float):
        params = [{'name': "output",
                   'details': {'shape': shape_full, 'dtype': dtype, 'negative': True}},
                  {'name': "multiplicand",
                   'details': {'shape': shape_full, 'dtype': dtype, 'negative': True}},
                  {'name': "multiplier",
                   'details': {'value': number_float}},
                  ]
        TestMathsMethods.universal_test("multiply_out_of_place", sut, params)

    @staticmethod
    def test_multiply_out_of_place_elementwise(sut, shape_full, dtype, dtype_mixed):
        params = [{'name': "output",
                   'details': {'shape': shape_full, 'dtype': dtype, 'negative': True}},
                  {'name': "multiplicand",
                   'details': {'shape': shape_full, 'dtype': dtype, 'negative': True}},
                  {'name': "multiplier",
                   'details': {'shape': shape_full, 'dtype': dtype_mixed, 'negative': True}},
                  ]
        TestMathsMethods.universal_test("multiply_out_of_place", sut, params)

    @staticmethod
    def test_power(sut, shape_full, dtype, number):
        params = [{'name': "output",
                   'details': {'shape': shape_full, 'dtype': dtype, 'negative': True}},
                  {'name': "exponent",
                   'details': {'value': number}},
                  ]
        TestMathsMethods.universal_test("power", sut, params)

    @staticmethod
    def test_subtract(sut, shape_full, dtype):
        params = [{'name': "output",
                  'details': {'shape': shape_full, 'dtype': dtype}},
                  {'name': "subtrahend",
                   'details': {'shape': shape_full, 'dtype': dtype}},
                  ]
        TestMathsMethods.universal_test("subtract", sut, params)

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
