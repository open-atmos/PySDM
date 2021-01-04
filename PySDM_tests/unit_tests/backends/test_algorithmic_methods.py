"""
Created at 20.04.2020
"""

import pytest
import numpy as np
# noinspection PyUnresolvedReferences
from PySDM_tests.unit_tests.backends.__parametrisation__ import \
    number_float, number, \
    shape_full, shape_1d, shape_2d, \
    natural_length, length, \
    order, pairs, \
    dtype_full, dtype, dtype_mixed
from PySDM_tests.unit_tests.backends.__parametrisation__ import backends
from .utils import universal_test, generate_is_first_in_pair, generate_data
from .__parametrisation__ import backend


# TODO: not implemented
@pytest.mark.skip()
@pytest.mark.parametrize('sut', backends)
class TestAlgorithmicMethods:

    @staticmethod
    def test_calculate_displacement(sut):
        assert False

    @staticmethod
    def test_coalescence(sut, shape_2d, order, natural_length, pairs):
        _, data = generate_data(sut, shape=(shape_2d[1],), dtype=float, factor=5)
        _, is_first_in_pair = generate_is_first_in_pair(sut, shape=(shape_2d[1],), pairs=pairs)
        backend.multiply(data, is_first_in_pair)
        gamma = np.floor(backend.to_ndarray(data))

        params = [{'name': "n",
                   'details': {'shape': (shape_2d[1],), 'dtype': int}},
                  {'name': "volume",
                   'details': {'shape': (shape_2d[1],), 'dtype': float}},
                  {'name': "idx",
                   'details': {'shape': shape_2d, 'order': order}},
                  {'name': "length",
                   'details': {'shape': shape_2d, 'length': natural_length}},
                  {'name': "intensive",
                   'details': {'shape': shape_2d, 'dtype': float}},
                  {'name': "extensive",
                   'details': {'shape': shape_2d, 'dtype': float}},
                  {'name': "gamma",
                   'details': {'array': gamma}},
                  {'name': "healthy",
                   'details': {'shape': (1,), 'dtype': int}}
                  ]
        universal_test("coalescence", sut, params)

    @staticmethod
    def test_compute_gamma(sut, shape_1d):
        params = [{'name': "prob",
                   'details': {'shape': shape_1d, 'dtype': float, 'seed': 44, 'factor': 5}},
                  {'name': "rand",
                   'details': {'shape': shape_1d, 'dtype': float, 'seed': 66, 'factor': 1}}
                  ]
        universal_test("compute_gamma", sut, params)

    @staticmethod
    def test_condensation(sut):
        assert False

    @staticmethod
    def test_flag_precipitated(sut):
        assert False

    @staticmethod
    def test_make_cell_caretaker(sut):
        assert False

    @staticmethod
    def test_moments(sut):
        assert False

    @staticmethod
    def test_normalize(sut):
        assert False

    @staticmethod
    @pytest.mark.parametrize('data_ndarray', [
        np.array([0] * 87),
        np.array([1, 0, 1, 0, 1, 1, 1, 1]),
        np.array([1, 1, 1, 1, 1, 0, 1, 0]),
        np.array([1] * 87)
    ])
    def test_remove_zeros(sut, data_ndarray, length, order):
        params = [{'name': "data",
                   'details': {'array': data_ndarray}},
                  {'name': "idx",
                   'details': {'shape': (data_ndarray.shape[0], ), 'order': order},
                   'checking': ['sorted', 'length_valid']},
                  {'name': "length",
                   'details': {'shape': (data_ndarray.shape[0], ), 'length': length}}
                  ]
        universal_test("remove_zeros", sut, params)


