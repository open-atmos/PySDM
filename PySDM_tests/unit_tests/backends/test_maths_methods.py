"""
Created at 20.03.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import pytest
# noinspection PyUnresolvedReferences
from PySDM_tests.unit_tests.backends.__parametrisation__ import \
    number_float, number, \
    shape_full, shape_1d, shape_2d, \
    dtype_full, dtype, dtype_mixed
from PySDM_tests.unit_tests.backends.__parametrisation__ import backend, backends
from .utils import universal_test, generate_data, generate_idx


@pytest.mark.parametrize('sut', backends)
class TestMathsMethods:

    @staticmethod
    def test_add(sut, shape_full, dtype):
        params = [{'name': "output",
                  'details': {'shape': shape_full, 'dtype': dtype}},
                  {'name': "addend",
                   'details': {'shape': shape_full, 'dtype': dtype}},
                  ]
        universal_test("add", sut, params)

    @staticmethod
    def test_column_modulo(sut, shape_2d, dtype_full):
        params = [{'name': "output",
                   'details': {'shape': shape_2d, 'dtype': int}},
                  {'name': "divisor",
                   'details': {'shape': (shape_2d[1],), 'dtype': int}},
                  ]
        universal_test("column_modulo", sut, params)

    @staticmethod
    def test_floor(sut, shape_1d):
        params = [{'name': "output",
                   'details': {'shape': shape_1d, 'dtype': float, 'negative': True}}
                  ]
        universal_test("floor", sut, params)

    @staticmethod
    def test_floor_out_of_place(sut, shape_2d):
        params = [{'name': "output",
                   'details': {'shape': shape_2d, 'dtype': int}},
                  {'name': "input_data",
                   'details': {'shape': shape_2d, 'dtype': float, 'negative': True}},
                  ]
        universal_test("floor_out_of_place", sut, params)

    @staticmethod
    def test_multiply_scalar(sut, shape_full, dtype, number_float):
        params = [{'name': "output",
                   'details': {'shape': shape_full, 'dtype': dtype, 'negative': True}},
                  {'name': "multiplier",
                   'details': {'value': number_float}},
                  ]
        universal_test("multiply", sut, params)

    @staticmethod
    def test_multiply_elementwise(sut, shape_full, dtype, dtype_mixed):
        params = [{'name': "output",
                   'details': {'shape': shape_full, 'dtype': dtype, 'negative': True}},
                  {'name': "multiplier",
                   'details': {'shape': shape_full, 'dtype': dtype_mixed, 'negative': True}},
                  ]
        universal_test("multiply", sut, params)

    @staticmethod
    def test_multiply_out_of_place_scalar(sut, shape_full, dtype, number_float):
        params = [{'name': "output",
                   'details': {'shape': shape_full, 'dtype': dtype, 'negative': True}},
                  {'name': "multiplicand",
                   'details': {'shape': shape_full, 'dtype': dtype, 'negative': True}},
                  {'name': "multiplier",
                   'details': {'value': number_float}},
                  ]
        universal_test("multiply_out_of_place", sut, params)

    @staticmethod
    def test_multiply_out_of_place_elementwise(sut, shape_full, dtype, dtype_mixed):
        params = [{'name': "output",
                   'details': {'shape': shape_full, 'dtype': dtype, 'negative': True}},
                  {'name': "multiplicand",
                   'details': {'shape': shape_full, 'dtype': dtype, 'negative': True}},
                  {'name': "multiplier",
                   'details': {'shape': shape_full, 'dtype': dtype_mixed, 'negative': True}},
                  ]
        universal_test("multiply_out_of_place", sut, params)

    @staticmethod
    def test_power(sut, shape_full, dtype, number):
        params = [{'name': "output",
                   'details': {'shape': shape_full, 'dtype': dtype, 'negative': True}},
                  {'name': "exponent",
                   'details': {'value': number}},
                  ]
        universal_test("power", sut, params)

    @staticmethod
    def test_subtract(sut, shape_full, dtype):
        params = [{'name': "output",
                  'details': {'shape': shape_full, 'dtype': dtype}},
                  {'name': "subtrahend",
                   'details': {'shape': shape_full, 'dtype': dtype}},
                  ]
        universal_test("subtract", sut, params)

    @staticmethod
    def test_urand(sut, shape_1d):
        # Arrange
        sut_data, data = generate_data(sut, shape_1d, float)
        sut_idx, idx = generate_idx(sut, shape_1d, 'asc')
        length = shape_1d[0]

        # Act
        sut.urand(sut_data)
        backend.urand(data)

        # Assert
        assert sut_data.shape == data.shape
        assert sut_data.dtype == data.dtype
        assert sut.amin(sut_data, sut_idx, length) >= 0
        assert sut.amax(sut_data, sut_idx, length) <= 1
