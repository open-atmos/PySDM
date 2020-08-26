"""
Created at 06.04.2020
"""

import pytest
# noinspection PyUnresolvedReferences
from PySDM_tests.unit_tests.backends.__parametrisation__ import \
    number_float, number, \
    shape_full, shape_1d, shape_2d, \
    natural_length, length, \
    order, pairs, \
    dtype_full, dtype, dtype_mixed
from PySDM_tests.unit_tests.backends.__parametrisation__ import backends
from .utils import universal_test


# TODO: not implemented
@pytest.mark.skip()
@pytest.mark.parametrize('sut', backends)
class TestAlgorithmicStepMethods:

    @staticmethod
    def test_amax_and_amin(sut, shape_1d, natural_length, order):
        params = [{'name': "row",
                   'details': {'shape': shape_1d, 'dtype': float}},
                  {'name': "idx",
                   'details': {'shape': shape_1d, 'order': order}},
                  {'name': "length",
                   'details': {'length': natural_length, 'shape': shape_1d}}
                  ]
        universal_test("amax", sut, params)
        universal_test("amin", sut, params)

    @staticmethod
    def test_cell_id(sut, shape_2d):
        params = [{'name': "cell_id",
                   'details': {'shape': (shape_2d[0],), 'dtype': int}},
                  {'name': "cell_origin",
                   'details': {'shape': shape_2d, 'dtype': int}},
                  {'name': "strides",
                   'details': {'shape': (1, shape_2d[1]), 'dtype': int}}
                  ]
        universal_test("cell_id", sut, params)

    @staticmethod
    def pair_params(shape_1d, pairs, order, length):
        params = [{'name': "data_out",
                   'details': {'shape': shape_1d, 'dtype': float}},
                  {'name': "data_in",
                   'details': {'shape': shape_1d, 'dtype': float}},
                  {'name': "is_first_in_pair",
                   'details': {'shape': shape_1d, 'pairs': pairs}},
                  {'name': "idx",
                   'details': {'shape': shape_1d, 'order': order}},
                  {'name': "length",
                   'details': {'shape': shape_1d, 'length': length}}
                  ]
        return params

    @staticmethod
    def test_distance_pair(sut, shape_1d, pairs, order, length):
        params = TestAlgorithmicStepMethods.pair_params(shape_1d, pairs, order, length)
        universal_test("distance_pair", sut, params)

    # TODO (FUTURE): better test cases
    @staticmethod
    def test_find_pairs(sut, shape_1d, pairs, order, length):
        cell_num = 3
        params = [{'name': "cell_start",
                   'details': {'shape': (cell_num,), 'dtype': int, 'factor': shape_1d[0]}},
                  {'name': "is_first_in_pair",
                   'details': {'shape': shape_1d, 'pairs': pairs}},
                  {'name': "cell_id",
                   'details': {'shape': shape_1d, 'dtype': int, 'factor': shape_1d[0]}},
                  {'name': "idx",
                   'details': {'shape': shape_1d, 'order': order}},
                  {'name': "length",
                   'details': {'shape': shape_1d, 'length': length}}
                  ]
        universal_test("find_pairs", sut, params)

    @staticmethod
    def test_max_pair(sut, shape_1d, pairs, length, order):
        params = TestAlgorithmicStepMethods.pair_params(shape_1d, pairs, order, length)
        params[1]['details']['dtype'] = int
        universal_test("max_pair", sut, params)

    @staticmethod
    def test_sum_pair(sut, shape_1d, pairs, length, order):
        params = TestAlgorithmicStepMethods.pair_params(shape_1d, pairs, order, length)
        universal_test("sum_pair", sut, params)

